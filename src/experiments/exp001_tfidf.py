"""EXP-001 — B1 TF-IDF + Cosine Similarity trên POJ-104.

Chạy: python -m src.experiments.exp001_tfidf --config configs/experiments/exp001.yaml
(vào repo root, venv đã activate).

Protocol:
- Fit TF-IDF CHỈ trên train samples (streaming, §6). Freeze rồi transform.
- Threshold chọn trên validation (tối ưu F1), freeze, đánh giá test đúng 1 lần (§11).
- Không regenerate dataset: artifact thiếu → fail loudly (§23); leakage != passed → abort (§24).
- Sparse toàn trình, không .toarray() full dataset (§17); batch inference (§16).
"""

import argparse
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
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.core.interfaces import CloneDetector
from src.data.access import stream_codes, stream_pairs
from src.evaluation.metrics import classification_metrics
from src.models.baseline.tfidf import code_tokenizer


# ---------------------------------------------------------------- validation
def validate_dataset(manifest_path: Path, pairs_dir: Path, processed_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Thiếu manifest: {manifest_path} (§23)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("leakage_passed", False):
        raise RuntimeError("ABORT EXP-001: leakage_passed != true (§24)")
    for split in ("train", "validation", "test"):
        p = pairs_dir / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Thiếu pair artifact: {p} (§23)")
    if not processed_path.exists():
        raise FileNotFoundError(f"Thiếu processed samples: {processed_path} (§23)")
    return manifest


# ---------------------------------------------------------------- vectorize
def build_vectorizer(max_features: int) -> TfidfVectorizer:
    # Tokenization (DEC-014): regex [A-Za-z_][A-Za-z0-9_]*|\\S, giữ nguyên case
    # (lowercase=False), không xử lý comment, không rename identifier.
    return TfidfVectorizer(
        tokenizer=code_tokenizer, token_pattern=None,
        lowercase=False, max_features=max_features,
    )


def fit_on_train(vectorizer: TfidfVectorizer, processed_path: Path) -> tuple[list[int], float]:
    """Fit trên train codes (stream). Trả về độ dài docs + thời gian fit."""
    t0 = time.perf_counter()
    lengths, n = [], 0

    def gen():
        for _, code in stream_codes(processed_path, "train"):
            lengths.append(len(code))
            yield code

    # sklearn fit tiêu thụ generator 1 lần; đếm n riêng sau transform.
    vectorizer.fit(gen())
    fit_time = time.perf_counter() - t0
    return lengths, fit_time


def transform_split(
    vectorizer: TfidfVectorizer, processed_path: Path, split: str
) -> tuple[dict[str, int], sparse.csr_matrix]:
    ids, rows = [], []
    t0 = time.perf_counter()
    mat = vectorizer.transform(
        _codes_and_ids(processed_path, split, ids)
    ).tocsr()
    dt = time.perf_counter() - t0
    return {sid: i for i, sid in enumerate(ids)}, mat, dt


def _codes_and_ids(processed_path: Path, split: str, ids: list):
    for sid, code in stream_codes(processed_path, split):
        ids.append(sid)
        yield code


# ---------------------------------------------------------------- scoring
def score_pairs(
    pairs_path: Path, id_to_index: dict[str, int], mat: sparse.csr_matrix,
    batch_size: int, collect_meta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Cosine trên sparse rows đã L2-normalize (= dot product).

    Chỉ giữ scores + labels (§16); per-pair meta chỉ thu cho split cần
    error analysis để khỏi giữ hàng trăm nghìn dicts vô ích.
    """
    scores, labels, meta = [], [], []
    t0 = time.perf_counter()
    for batch in stream_pairs(pairs_path, batch_size):
        ia = [id_to_index[p["code_a_id"]] for p in batch]
        ib = [id_to_index[p["code_b_id"]] for p in batch]
        a, b = mat[ia], mat[ib]
        s = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        s = np.clip(s, 0.0, 1.0)
        scores.append(s)
        labels.extend(p["label"] for p in batch)
        if collect_meta:
            for p, v in zip(batch, s):
                meta.append({"pair_id": p["pair_id"], "label": p["label"], "score": float(v)})
    dt = time.perf_counter() - t0
    return np.concatenate(scores), np.array(labels).ravel(), meta, dt


def select_threshold(val_scores: np.ndarray, val_labels: np.ndarray, step: float = 0.01) -> dict:
    """Chọn threshold trên VALIDATION duy nhất, tối ưu F1 (§11)."""
    best = {"threshold": 0.5, "f1": -1.0}
    for t in np.arange(step, 1.0, step):
        m = classification_metrics(
            val_labels.tolist(), val_scores.tolist(), threshold=float(t)
        )
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


# ---------------------------------------------------------------- experiment
class TfidfExperiment(CloneDetector):
    """B1 qua abstraction CloneDetector (§9). Chạy batch/streaming."""

    def __init__(self, vectorizer: TfidfVectorizer, id_to_index, mat, threshold: float = 0.5):
        self.vectorizer = vectorizer
        self.id_to_index = id_to_index
        self.mat = mat
        self.threshold = threshold

    def predict_proba(self, pairs: list) -> list[float]:
        ia = [self.id_to_index[p["code_a_id"]] for p in pairs]
        ib = [self.id_to_index[p["code_b_id"]] for p in pairs]
        s = np.asarray(self.mat[ia].multiply(self.mat[ib]).sum(axis=1)).ravel()
        return [float(np.clip(v, 0.0, 1.0)) for v in s]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(config_path: Path, manifest_path: Path) -> dict:
    import sklearn

    meta = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "device": "cpu",
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    try:
        import torch

        meta["versions"]["torch"] = torch.__version__
        meta["versions"]["transformers"] = __import__("transformers").__version__
    except ImportError:
        pass
    try:
        import subprocess

        meta["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "n/a (not a git repo)"
    except Exception:
        meta["git"] = "n/a"
    return meta


def run_experiment(cfg: dict, config_path: Path, out_dir: Path) -> dict:
    set_global_seed(cfg["training"]["seed"])
    tracemalloc.start()
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    manifest = validate_dataset(manifest_path, pairs_dir, Path(cfg["dataset"]["processed"]))
    processed = Path(cfg["dataset"]["processed"])
    batch = cfg["inference"]["pair_batch_size"]

    vec = build_vectorizer(cfg["model"]["max_features"])
    train_lengths, fit_time = fit_on_train(vec, processed)

    mats, t_transform = {}, 0.0
    for split in ("train", "validation", "test"):
        idx, mat, dt = transform_split(vec, processed, split)
        mats[split] = (idx, mat)
        t_transform += dt

    scores, t_score = {}, 0.0
    for split in ("train", "validation", "test"):
        idx, mat = mats[split]
        s, y, meta, dt = score_pairs(
            pairs_dir / f"{split}.jsonl", idx, mat, batch,
            collect_meta=(split == "test"),
        )
        scores[split] = (s, y, meta)
        t_score += dt

    thr = select_threshold(*scores["validation"][:2], step=cfg["threshold"]["grid_step"])
    threshold = thr["threshold"]
    results = {
        split: split_metrics(s, y, threshold) for split, (s, y, _) in scores.items()
    }

    # Error cases top-k (§15): FP score cao nhất, FN score thấp nhất.
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

    # Artifacts (§19)
    (out_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copy(manifest_path, out_dir / "manifest.json")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    import csv

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "TP", "TN", "FP", "FN"]
        w.writerow(["split"] + keys)
        for split, m in results.items():
            w.writerow([split] + [m[k] for k in keys])
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
    vocab = len(vec.vocabulary_)
    nnz = sum(mat.nnz for _, mat in mats.values())
    n_cells = sum(mat.shape[0] * mat.shape[1] for _, mat in mats.values())
    (out_dir / "tfidf_stats.json").write_text(json.dumps({
        "vocabulary_size": vocab,
        "n_train_docs": len(train_lengths),
        "doc_len_chars": {
            "mean": float(np.mean(train_lengths)),
            "median": float(np.median(train_lengths)),
            "max": int(np.max(train_lengths)),
        },
        "sparsity": 1.0 - nnz / n_cells,
        "fit_time_s": fit_time,
        "transform_time_s": t_transform,
        "scoring_time_s": t_score,
        "total_time_s": total_time,
        "peak_python_alloc_mb": peak / 1e6,
    }, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata(config_path, manifest_path), indent=2), encoding="utf-8")

    summary = {
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
