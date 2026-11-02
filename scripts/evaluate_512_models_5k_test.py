#!/usr/bin/env python3
"""Chạy đánh giá M1 và M2 bản 512 token trên 5.000 cặp Test BigCloneBench."""

import json
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIRS_CACHE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
RESULTS_512_FILE = ROOT / "results" / "512" / "bcb_test_5k_512_results.json"
RESULTS_ALL_FILE = ROOT / "results" / "bcb_test_5k_results.json"


def get_5k_test_pairs() -> list[dict]:
    pairs = []
    with open(PAIRS_CACHE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


def find_best_threshold(scores, labels, fixed_th=None) -> tuple[float, dict]:
    if fixed_th is not None:
        preds = [1 if s >= fixed_th else 0 for s in scores]
        return fixed_th, compute_metrics(labels, preds)

    best_th, best_f1, best_m = 0.5, -1.0, None
    for th in np.linspace(0.01, 0.99, 99):
        preds = [1 if s >= th else 0 for s in scores]
        m = compute_metrics(labels, preds)
        if m["f1"] > best_f1:
            best_f1, best_th, best_m = m["f1"], th, m
    return best_th, best_m


def eval_model(model_name: str, hf_id: str, ckpt_path: str, max_length: int = 512, fixed_th=None):
    print(f"\n========================================================")
    print(f"Đang đánh giá {model_name} (max_length={max_length}) trên 5.000 mẫu Test...")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {DEVICE}")

    pairs = get_5k_test_pairs()
    labels = [p["label"] for p in pairs]
    codes_1 = [p["func1"] for p in pairs]
    codes_2 = [p["func2"] for p in pairs]

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, dropout=0.1).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = []
    bs = 32  # batch size 32 an toàn với max_len 512 trên RTX 5060 Laptop (8GB)

    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            b1 = tokenizer(codes_1[i : i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(codes_2[i : i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
            logits = model(b1, b2)
            probs = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, list):
                scores.extend(probs)
            else:
                scores.append(probs)
            if (i // bs) % 25 == 0:
                print(f"  Processed {min(i + bs, len(pairs))}/{len(pairs)} pairs...", flush=True)

    # 1. Đo theo ngưỡng tối ưu tốt nhất tìm được trên 5k test
    opt_th, opt_m = find_best_threshold(scores, labels, fixed_th=None)
    # 2. Đo theo ngưỡng fixed_th (nếu có từ quá trình train/val)
    if fixed_th is not None:
        f_th, f_m = find_best_threshold(scores, labels, fixed_th=fixed_th)
    else:
        f_th, f_m = opt_th, opt_m

    res = {
        "fixed": {
            "th": round(float(f_th), 4),
            "tp": f_m["tp"], "tn": f_m["tn"], "fp": f_m["fp"], "fn": f_m["fn"],
            "f1": round(f_m["f1"] * 100, 2), "prec": round(f_m["prec"] * 100, 2),
            "rec": round(f_m["rec"] * 100, 2), "acc": round(f_m["acc"] * 100, 2),
        },
        "optimal": {
            "th": round(float(opt_th), 4),
            "tp": opt_m["tp"], "tn": opt_m["tn"], "fp": opt_m["fp"], "fn": opt_m["fn"],
            "f1": round(opt_m["f1"] * 100, 2), "prec": round(opt_m["prec"] * 100, 2),
            "rec": round(opt_m["rec"] * 100, 2), "acc": round(opt_m["acc"] * 100, 2),
        }
    }
    print(f"Kết quả {model_name}:")
    print(f"  [Ngưỡng cố định {f_th}]: F1={res['fixed']['f1']}%, Acc={res['fixed']['acc']}%, Prec={res['fixed']['prec']}%, Rec={res['fixed']['rec']}%")
    print(f"  [Ngưỡng tối ưu {opt_th:.2f}]: F1={res['optimal']['f1']}%, Acc={res['optimal']['acc']}%, Prec={res['optimal']['prec']}%, Rec={res['optimal']['rec']}%")
    return res


def main():
    m1_res = eval_model(
        model_name="M1_CodeBERT_512",
        hf_id="microsoft/codebert-base",
        ckpt_path="results/512/exp_bcb_002/best_checkpoint.pt",
        max_length=512,
        fixed_th=0.88,
    )

    m2_res = eval_model(
        model_name="M2_GraphCodeBERT_512",
        hf_id="microsoft/graphcodebert-base",
        ckpt_path="results/512/exp_bcb_003/best_checkpoint.pt",
        max_length=512,
        fixed_th=0.88,
    )

    all_res = {
        "M1_512": m1_res,
        "M2_512": m2_res,
    }

    RESULTS_512_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_512_FILE, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\nĐã lưu kết quả 512 vào {RESULTS_512_FILE}")

    # Đồng thời cập nhật vào kết quả chung kết hợp
    if RESULTS_ALL_FILE.exists():
        with open(RESULTS_ALL_FILE, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        full_data["M1_512"] = m1_res["fixed"]
        full_data["M1_512_opt"] = m1_res["optimal"]
        full_data["M2_512"] = m2_res["fixed"]
        full_data["M2_512_opt"] = m2_res["optimal"]
        with open(RESULTS_ALL_FILE, "w", encoding="utf-8") as f:
            json.dump(full_data, f, indent=2)
        print(f"Đã đồng bộ cập nhật vào {RESULTS_ALL_FILE}")


if __name__ == "__main__":
    main()
