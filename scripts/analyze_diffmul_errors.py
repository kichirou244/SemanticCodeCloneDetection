#!/usr/bin/env python3
"""Phân tích lỗi (FP/FN) của M1/M2 DiffMul trên 5.000 cặp BigCloneBench.

Với mỗi cặp, script tính thêm độ tương tự cosine giữa hai vector CLS
``cos(hA, hB)`` để quan sát quan hệ giữa "độ gần nhau trong không gian biểu diễn"
và quyết định đúng/sai của mô hình.

Đầu ra (mặc định trong ``results/error_analysis_diffmul/``):
  - ``{model}_all_scores.jsonl`` : toàn bộ 5.000 cặp (prob, cosine, dự đoán)
  - ``{model}_errors.csv``       : danh sách FP + FN kèm cosine
  - ``{model}_errors.json``      : thống kê tổng hợp + danh sách lỗi
  - ``error_analysis_diffmul.md``: báo cáo Markdown để đọc trực tiếp
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier  # noqa: E402

PAIRS_FILE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
DEFAULT_OUT_DIR = ROOT / "results" / "error_analysis_diffmul"

MODEL_SPECS = {
    "m1": {
        "label": "M1 CodeBERT DiffMul",
        "ckpt": ROOT / "results" / "exp_m1_diffmul" / "best_checkpoint.pt",
    },
    "m2": {
        "label": "M2 GraphCodeBERT DiffMul",
        "ckpt": ROOT / "results" / "exp_m2_diffmul" / "best_checkpoint.pt",
    },
}


def load_pairs(path: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def method_name(code: str) -> str:
    pattern = r"(?:public|protected|private|static|final|synchronized|\s)+[\w<>\[\],.?]+\s+(\w+)\s*\("
    for line in code.splitlines()[:8]:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("*") or line.startswith("/*"):
            continue
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return "unknown"


def preview(code: str, max_chars: int = 240) -> str:
    text = " ".join(code.split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def cosine_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(values.std()), 4),
        "min": round(float(values.min()), 4),
        "q25": round(float(np.quantile(values, 0.25)), 4),
        "q75": round(float(np.quantile(values, 0.75)), 4),
        "max": round(float(values.max()), 4),
    }


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
    }


@torch.no_grad()
def infer_model(
    spec: dict[str, Any],
    pairs: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    use_amp: bool = True,
) -> dict[str, Any]:
    ckpt = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model_cfg = cfg["model"]
    hf_id = model_cfg["hf_id"]
    dropout = float(model_cfg.get("dropout", 0.1))
    pair_arch = model_cfg.get("pair_architecture", "diff_mul")
    threshold = float(ckpt.get("best_threshold", 0.5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{spec['label']}] nạp {hf_id} (dropout={dropout}, arch={pair_arch}) trên {device} ...", flush=True)
    if spec.get("attn_implementation") in (None, "default"):
        model = TransformerPairClassifier(hf_id=hf_id, dropout=dropout, pair_architecture=pair_arch).to(device)
    else:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(hf_id, attn_implementation=spec["attn_implementation"])
        model = TransformerPairClassifier(
            hf_id=hf_id,
            dropout=dropout,
            encoder=encoder,
            pair_architecture=pair_arch,
        ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [cảnh báo] thiếu {len(missing)} khóa: {missing[:3]}", flush=True)
    if unexpected:
        print(f"  [cảnh báo] thừa {len(unexpected)} khóa: {unexpected[:3]}", flush=True)
    model.eval()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)

    probs: list[float] = []
    cosines: list[float] = []
    total = len(pairs)
    for start in range(0, total, batch_size):
        batch = pairs[start : start + batch_size]
        codes_a = [p["func1"].replace("\r\n", "\n").replace("\r", "\n").strip() for p in batch]
        codes_b = [p["func2"].replace("\r\n", "\n").replace("\r", "\n").strip() for p in batch]
        inputs_a = tok(codes_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        inputs_b = tok(codes_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            ha = model.encode_side(inputs_a)
            hb = model.encode_side(inputs_b)
            if pair_arch in ("diff_mul", "absdiff_mul", "diff_prod", "hadamard_diff", "diff_hadamard"):
                pair_repr = torch.cat([torch.abs(ha - hb), ha * hb], dim=-1)
            else:
                pair_repr = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
            logits = model.head(pair_repr).squeeze(-1)
        probs.extend(torch.sigmoid(logits).float().cpu().tolist())
        # Cosine luôn tính ở fp32 để ổn định số học
        cosines.extend(F.cosine_similarity(ha.float(), hb.float(), dim=-1).cpu().tolist())

        done = min(start + batch_size, total)
        if done % (batch_size * 10) == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    labels = np.array([int(p["label"]) for p in pairs], dtype=int)
    prob_arr = np.array(probs, dtype=np.float64)
    cos_arr = np.array(cosines, dtype=np.float64)
    preds = (prob_arr >= threshold).astype(int)
    metrics = compute_metrics(labels, prob_arr, threshold)

    # Kiểm tra chéo với test_metrics.json đã ghi nhận trước đó
    recorded_path = spec["ckpt"].parent / "test_metrics.json"
    recorded = None
    if recorded_path.exists():
        recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
        if len(pairs) == 5000:
            same = all(metrics[k] == recorded[k] for k in ("tp", "tn", "fp", "fn"))
            print(
                f"  Đối chiếu test_metrics.json: {'KHỚP' if same else 'LỆCH'} "
                f"(tính lại TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} | "
                f"ghi nhận TP={recorded.get('tp')} TN={recorded.get('tn')} FP={recorded.get('fp')} FN={recorded.get('fn')})",
                flush=True,
            )
        else:
            print("  Bỏ qua đối chiếu test_metrics.json vì đang chạy giới hạn số cặp.", flush=True)

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        code1, code2 = pair["func1"], pair["func2"]
        rows.append(
            {
                "model": spec["key"],
                "model_label": spec["label"],
                "index": idx,
                "pair_id": str(pair.get("id", "")),
                "id1": str(pair.get("id1", "")),
                "id2": str(pair.get("id2", "")),
                "label": int(labels[idx]),
                "prob": round(float(prob_arr[idx]), 6),
                "pred": int(preds[idx]),
                "cosine": round(float(cos_arr[idx]), 6),
                "threshold": threshold,
                "len1": len(code1),
                "len2": len(code2),
                "name1": method_name(code1),
                "name2": method_name(code2),
                "func1_preview": preview(code1),
                "func2_preview": preview(code2),
            }
        )

    quadrant = {
        "TP": (labels == 1) & (preds == 1),
        "TN": (labels == 0) & (preds == 0),
        "FP": (labels == 0) & (preds == 1),
        "FN": (labels == 1) & (preds == 0),
    }
    cos_by_quadrant = {name: cosine_stats(cos_arr[mask]) for name, mask in quadrant.items()}
    prob_by_quadrant = {name: cosine_stats(prob_arr[mask]) for name, mask in quadrant.items()}

    errors = [r for r in rows if r["pred"] != r["label"]]
    fp_rows = sorted([r for r in errors if r["label"] == 0], key=lambda r: r["prob"], reverse=True)
    fn_rows = sorted([r for r in errors if r["label"] == 1], key=lambda r: r["prob"])

    corr = {}
    for label_value in (0, 1):
        mask = labels == label_value
        if mask.sum() > 1:
            corr[str(label_value)] = round(float(np.corrcoef(cos_arr[mask], prob_arr[mask])[0, 1]), 4)
    corr["all"] = round(float(np.corrcoef(cos_arr, prob_arr)[0, 1]), 4)

    return {
        "key": spec["key"],
        "label": spec["label"],
        "checkpoint": str(spec["ckpt"].resolve()),
        "pairs_file": spec.get("pairs_file", ""),
        "hf_id": hf_id,
        "threshold": threshold,
        "metrics": metrics,
        "recorded_metrics": recorded,
        "rows": rows,
        "errors": errors,
        "fp_rows": fp_rows,
        "fn_rows": fn_rows,
        "cos_by_quadrant": cos_by_quadrant,
        "prob_by_quadrant": prob_by_quadrant,
        "cos_prob_correlation": corr,
        "n_pairs": len(rows),
        "n_errors": len(errors),
    }


def write_outputs(result: dict[str, Any], out_dir: Path) -> None:
    key = result["key"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{key}_all_scores.jsonl", "w", encoding="utf-8") as f:
        for row in result["rows"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "model_label",
        "error_type",
        "index",
        "pair_id",
        "id1",
        "id2",
        "label",
        "prob",
        "cosine",
        "threshold",
        "len1",
        "len2",
    ]
    with open(out_dir / f"{key}_errors.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for error_type, rows in (("FP", result["fp_rows"]), ("FN", result["fn_rows"])):
            for index, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "model_label": row["model_label"],
                        "error_type": error_type,
                        "index": index,
                        "pair_id": row["pair_id"],
                        "id1": row["id1"],
                        "id2": row["id2"],
                        "label": row["label"],
                        "prob": row["prob"],
                        "cosine": row["cosine"],
                        "threshold": row["threshold"],
                        "len1": row["len1"],
                        "len2": row["len2"],
                    }
                )

    payload = {
        "model": key,
        "label": result["label"],
        "checkpoint": result["checkpoint"],
        "pairs_file": result["pairs_file"],
        "hf_id": result["hf_id"],
        "threshold": result["threshold"],
        "metrics": result["metrics"],
        "recorded_metrics": result["recorded_metrics"],
        "n_pairs": result["n_pairs"],
        "n_errors": result["n_errors"],
        "cosine_by_quadrant": result["cos_by_quadrant"],
        "prob_by_quadrant": result["prob_by_quadrant"],
        "cos_prob_correlation": result["cos_prob_correlation"],
        "false_positives": result["fp_rows"],
        "false_negatives": result["fn_rows"],
    }
    with open(out_dir / f"{key}_errors.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_markdown(results: list[dict[str, Any]], out_dir: Path) -> Path:
    lines: list[str] = []
    lines.append("# Phân tích lỗi M1/M2 DiffMul kèm độ tương tự cosine\n")
    lines.append(
        "Cosine được tính giữa hai vector CLS `hA`, `hB` (trước tầng tương tác DiffMul): "
        "`cos(hA, hB)`. Giá trị càng gần 1 nghĩa là hai hàm càng gần nhau trong không gian biểu diễn.\n"
    )
    lines.append(
        "Cột `prob` là xác suất dự đoán cặp là clone, **khác với cosine**. Cách tính:\n\n"
        "```python\n"
        "# src/models/transformer_pair.py :: TransformerPairClassifier.forward()\n"
        "ha = encoder(inputs_a).last_hidden_state[:, 0, :]   # vector CLS của hàm A\n"
        "hb = encoder(inputs_b).last_hidden_state[:, 0, :]   # vector CLS của hàm B\n"
        "diff = torch.abs(ha - hb)\n"
        "prod = ha * hb                                      # Hadamard product\n"
        "pair = torch.cat([diff, prod], dim=-1)              # 1536 chiều (2 x 768)\n"
        "logits = head(pair).squeeze(-1)                     # Linear->Tanh->Dropout->Linear\n"
        "\n"
        "# src/experiments/bcb_runner.py / scripts/analyze_diffmul_errors.py\n"
        "prob = torch.sigmoid(logits)                        # giá trị nằm trong [0, 1]\n"
        "pred = (prob >= threshold).astype(int)              # M1: 0.12 | M2: 0.14\n"
        "```\n\n"
        "Lúc huấn luyện dùng `nn.BCEWithLogitsLoss()` nên sigmoid nằm bên trong loss; "
        "lúc inference mới lấy `sigmoid(logits)` thành cột `prob`. Ngưỡng được chọn từ F1 trên "
        "tập validation. `prob` là score chưa hiệu chỉnh (not calibrated) — không hiểu là xác suất "
        "thống kê, chỉ dùng để so ngưỡng. **Cosine không tham gia quyết định phân loại.**\n"
    )

    for res in results:
        m = res["metrics"]
        lines.append(f"## {res['label']}\n")
        lines.append(f"- Checkpoint: `{res['checkpoint']}`")
        lines.append(f"- Tập test: `{res['pairs_file']}`")
        lines.append(f"- Ngưỡng: `{res['threshold']}`")
        lines.append(
            f"- Kết quả: TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, "
            f"F1={m['f1']:.4f}, Accuracy={m['accuracy']:.4f}\n"
        )
        lines.append("### Thống kê cosine theo nhóm dự đoán\n")
        lines.append("| Nhóm | Số cặp | Cosine trung bình | Trung vị | Q25 | Q75 | Min | Max |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for quad in ("TP", "TN", "FP", "FN"):
            stats = res["cos_by_quadrant"].get(quad, {})
            if not stats:
                continue
            lines.append(
                f"| {quad} | {stats['count']} | {fmt(stats['mean'])} | {fmt(stats['median'])} | "
                f"{fmt(stats['q25'])} | {fmt(stats['q75'])} | {fmt(stats['min'])} | {fmt(stats['max'])} |"
            )
        lines.append("")
        lines.append(
            "Tương quan Pearson giữa cosine và xác suất dự đoán: "
            f"toàn bộ = {res['cos_prob_correlation'].get('all')}, "
            f"non-clone = {res['cos_prob_correlation'].get('0')}, "
            f"clone = {res['cos_prob_correlation'].get('1')}.\n"
        )

        for err_type, rows, title in (
            ("FP", res["fp_rows"], "False Positive (dự đoán Clone, nhãn Non-clone)"),
            ("FN", res["fn_rows"], "False Negative (dự đoán Non-clone, nhãn Clone)"),
        ):
            lines.append(f"### {title} — tổng {len(rows)} ca\n")
            lines.append("| # | Pair ID | id1 | id2 | prob | cosine | Tên hàm A | Tên hàm B |")
            lines.append("| ---: | --- | --- | --- | ---: | ---: | --- | --- |")
            for rank, row in enumerate(rows[:20], start=1):
                lines.append(
                    f"| {rank} | {row['pair_id']} | {row['id1']} | {row['id2']} | "
                    f"{row['prob']:.4f} | {row['cosine']:.4f} | `{row['name1']}` | `{row['name2']}` |"
                )
            if len(rows) > 20:
                lines.append(f"\n*(Chỉ hiển thị 20/{len(rows)} ca; danh sách đầy đủ trong `{res['key']}_errors.csv`.)*\n")
            else:
                lines.append("")

    md_path = out_dir / "error_analysis_diffmul.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Phân tích FP/FN của M1/M2 DiffMul kèm cosine")
    parser.add_argument("--models", nargs="+", default=["m1", "m2"], choices=["m1", "m2"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pairs-file", type=Path, default=PAIRS_FILE)
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N cặp đầu (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--attn-implementation",
        default="default",
        choices=["default", "eager", "sdpa", "flash_attention_2"],
        help="Backend attention khi nạp encoder (eager giúp tái lập gần với transformers 4.x)",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "fp32"],
        help="fp16 = autocast giống lúc evaluate trên Kaggle; fp32 = suy luận toàn phần",
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    pairs = load_pairs(args.pairs_file)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"Đã nạp {len(pairs)} cặp từ {args.pairs_file}", flush=True)

    results: list[dict[str, Any]] = []
    for key in args.models:
        spec = dict(MODEL_SPECS[key])
        spec["key"] = key
        spec["attn_implementation"] = args.attn_implementation
        spec["pairs_file"] = str(args.pairs_file)
        result = infer_model(
            spec,
            pairs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_amp=args.precision == "fp16",
        )
        write_outputs(result, args.out_dir)
        results.append(result)

        m = result["metrics"]
        cos_means = {
            quad: result["cos_by_quadrant"].get(quad, {}).get("mean")
            for quad in ("TP", "TN", "FP", "FN")
        }
        cos_text = ", ".join(
            f"{quad}={value:.4f}" if value is not None else f"{quad}=n/a"
            for quad, value in cos_means.items()
        )
        print(
            f">>> {result['label']}: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']} "
            f"F1={m['f1']:.4f} | cosine TB: {cos_text}",
            flush=True,
        )

    if len(results) == len(args.models):
        md_path = write_markdown(results, args.out_dir)
        print(f"\nĐã ghi báo cáo: {md_path.resolve()}", flush=True)
        print(f"Thư mục kết quả: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
