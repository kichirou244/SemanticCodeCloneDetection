"""EXP-006 — B1 MOSS (winnowing fingerprinting) trên POJ-104.

Chạy: python -m src.experiments.exp006_moss --config configs/exp006_moss_poj104.yaml
(vào repo root, venv đã activate).

Vì sao MOSS thay TF-IDF làm baseline (RQ1/B1): MOSS là công cụ phát hiện đạo văn
chuẩn của Stanford. Dịch vụ thật cần tài khoản + upload code lên mạng; bản local
thể hiện đúng thuật toán fingerprinting (winnowing, k=50, w=100) trong
`src/models/baseline/moss.py`. Không có khâu fit — kết quả tất định trên nội dung.

Protocol (giữ nguyên từ EXP-001):
- Threshold duy nhất chọn trên validation, tối ưu F1 (§11), freeze, eval test 1 lần.
- Không regenerate dataset; artifact thiếu → fail (§23); leakage != passed → abort (§24).
- Artifacts đồng dạng EXP-001 để các tool đọc metrics thống nhất (§19).
"""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from sklearn.metrics import confusion_matrix

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.data.access import stream_codes, stream_pairs
from src.evaluation.metrics import classification_metrics
from src.models.baseline.moss import fingerprints, similarity

DEFAULT_K = 50
DEFAULT_W = 100


def validate_dataset(manifest_path: Path, pairs_dir: Path, processed_path: Path) -> dict:
    """Giống EXP-001: artifacts thiếu → fail loudly, leakage chưa pass → abort."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Thiếu manifest: {manifest_path} (§23)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("leakage_passed", False):
        raise RuntimeError("ABORT EXP-006: leakage_passed != true (§24)")
    for split in ("train", "validation", "test"):
        p = pairs_dir / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Thiếu pair artifact: {p} (§23)")
    if not processed_path.exists():
        raise FileNotFoundError(f"Thiếu processed samples: {processed_path} (§23)")
    return manifest


def load_codes(processed_path: Path, split: str, limit_chars: int = 20000) -> dict[str, str]:
    """Giữ code của một split trong RAM — POJ-104 nhỏ, fingerprint hoá nhanh."""
    codes: dict[str, str] = {}
    for sample_id, code in stream_codes(processed_path, split):
        codes[sample_id] = code[:limit_chars]
    return codes


def precompute(codes: dict[str, str], k: int, w: int) -> dict[str, list[int]]:
    """Fingerprint mọi code một lần rồi so cặp trên fingerprint (không lặp lại)."""
    return {sid: fingerprints(code, k=k, w=w) for sid, code in codes.items()}


def score_pairs(
    pairs_path: Path, fp_map: dict[str, list[int]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Containment fingerprint cho từng pair (tái dùng `similarity` của core MOSS).

    Dùng chính hàm `similarity` của `src/models/baseline/moss.py` để exp này dùng
    đúng một thước đo với demo scan — % trong bảng quét thư mục phải bằng %
    nhìn thấy trong báo cáo.
    """
    scores: list[float] = []
    labels: list[int] = []
    meta: list[dict] = []
    for batch in stream_pairs(pairs_path):
        for pair in batch:
            sim = similarity(fp_map[pair["code_a_id"]], fp_map[pair["code_b_id"]])
            scores.append(sim)
            labels.append(pair["label"])
            meta.append({
                "pair_id": pair["pair_id"], "label": pair["label"],
                "score": float(sim),
                "n_fp_a": len(fp_map[pair["code_a_id"]]),
                "n_fp_b": len(fp_map[pair["code_b_id"]]),
            })
    return np.array(scores), np.array(labels, dtype=np.int64), meta


def select_threshold(val_scores: np.ndarray, val_labels: np.ndarray, step: float = 0.01) -> dict:
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.arange(step, 1.0, step):
        m = classification_metrics(val_labels.tolist(), val_scores.tolist(), threshold=float(t))
        if m["f1"] > best["f1"]:
            best = {"threshold": float(t), "f1": m["f1"]}
    best["method"] = "validation_f1"
    best["split"] = "validation"
    return best


def split_metrics(scores, labels, threshold) -> dict:
    m = classification_metrics(labels.tolist(), scores.tolist(), threshold=threshold)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    m.update({
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    })
    return m


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(config_path: Path, manifest_path: Path) -> dict:
    meta = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "device": "cpu",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "versions": {"numpy": np.__version__},
    }
    try:
        import subprocess

        meta["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "n/a (not a git repo)"
    except Exception:  # noqa: BLE001
        meta["git"] = "n/a"
    return meta


def run_experiment(cfg: dict, config_path: Path, out_dir: Path) -> dict:
    set_global_seed(cfg["experiment"]["seed"])
    tracemalloc.start()
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    k = int(cfg["model"].get("k", DEFAULT_K))
    w = int(cfg["model"].get("w", DEFAULT_W))

    t_fp, scores = 0.0, {}
    for split in ("train", "validation", "test"):
        t0 = time.perf_counter()
        codes = load_codes(processed, split)
        fp_map = precompute(codes, k, w)
        t_fp += time.perf_counter() - t0
        s, y, meta = score_pairs(pairs_dir / f"{split}.jsonl", fp_map)
        scores[split] = (s, y, meta)

    thr = select_threshold(*scores["validation"][:2], step=cfg["threshold"]["grid_step"])
    threshold = thr["threshold"]
    results = {
        split: split_metrics(s, y, threshold) for split, (s, y, _) in scores.items()
    }

    top_k = cfg["error_analysis"]["top_k"]
    _, _, test_meta = scores["test"]
    fp = sorted([m for m in test_meta if m["label"] == 0 and m["score"] >= threshold],
                key=lambda m: -m["score"])[:top_k]
    fn = sorted([m for m in test_meta if m["label"] == 1 and m["score"] < threshold],
                key=lambda m: m["score"])[:top_k]
    for m in fp:
        m.update({"prediction": 1, "error_type": "false_positive"})
    for m in fn:
        m.update({"prediction": 0, "error_type": "false_negative"})

    total_time = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy(manifest_path, out_dir / "manifest.json")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "TP", "TN", "FP", "FN"]
        wcsv.writerow(["split"] + keys)
        for split, m in results.items():
            wcsv.writerow([split] + [m[k] for k in keys])
    (out_dir / "threshold.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps({s: {k: m[k] for k in ("TP", "TN", "FP", "FN")} for s, m in results.items()}, indent=2),
        encoding="utf-8",
    )
    with open(out_dir / "error_cases.jsonl", "w", encoding="utf-8") as f:
        for m in fp + fn:
            f.write(json.dumps(m) + "\n")
    with open(out_dir / "predictions_test_scores.jsonl", "w", encoding="utf-8") as f:
        for m in test_meta:
            f.write(json.dumps(m) + "\n")
    n_fp_stats = {
        split: {
            "min": int(float(np.min(s))) if len(s) else 0,
            "mean": float(np.mean(s)) if len(s) else 0.0,
            "max": int(float(np.max(s))) if len(s) else 0,
            "n_pairs": int(len(s)),
        }
        for split, (s, _, _) in scores.items()
    }
    (out_dir / "moss_stats.json").write_text(json.dumps({
        "k": k, "w": w, "fingerprint_pairs": n_fp_stats,
        "fingerprint_time_s": round(t_fp, 3),
        "threshold": threshold,
        "total_time_s": total_time,
        "peak_python_alloc_mb": peak / 1e6,
    }, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata(config_path, manifest_path), indent=2), encoding="utf-8")

    summary = {
        "baseline": "moss_winnowing", "k": k, "w": w,
        "threshold": threshold, "validation_f1_at_thr": thr["f1"],
        "test": results["test"], "total_time_s": total_time,
        "out_dir": str(out_dir),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg, Path(args.config), Path(cfg["output"]["directory"]))


if __name__ == "__main__":
    main()