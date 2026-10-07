"""Audit bổ trợ (complementarity) M2 GraphCodeBERT x M4 CPG-GNN trên POJ-104.

Chạy:
  python scripts/audit_m2_m4_complementarity.py
  python scripts/audit_m2_m4_complementarity.py --m2-result results/exp003_graphcodebert_poj104 --m4-result results/exp005_cpg_gnn_poj104 --output results/audit_m2_m4.json

Nội dung:
  1. So sánh nhị phân tại ngưỡng ghim (validation_f1) giữa 2 model — agreement + kappa.
  2. Bảng 2x2 "ai đúng ai sai" theo cặp.
  3. Số lỗi của M2 mà M4 sửa được (corrected) vs số đúng của M2 mà M4 làm hỏng (broken)
     — tín hiệu mức quyết định cho biết hybrid có cửa hay không.
  4. Fusion bằng logistic trên 2 score (fit trên validation, đánh giá trên test)
     — giữ đúng protocol, không chạm test khi chọn tham số.
  5. Kết luận: so F1 test của fusion với max(M2, M4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score


def load_scores(run_dir: Path, split: str) -> dict[str, tuple[int, float]]:
    path = run_dir / "predictions" / f"{split}_scores.jsonl"
    out: dict[str, tuple[int, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            out[str(r["pair_id"])] = (int(r["label"]), float(r["score"]))
    return out


def load_threshold(run_dir: Path) -> float:
    return float(json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"])


def prf(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2-result", default="results/exp003_graphcodebert_poj104")
    ap.add_argument("--m4-result", default="results/exp005_cpg_gnn_poj104")
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    m2_dir, m4_dir = Path(args.m2_result), Path(args.m4_result)
    d2, d4 = load_scores(m2_dir, args.split), load_scores(m4_dir, args.split)
    t2, t4 = load_threshold(m2_dir), load_threshold(m4_dir)

    common = sorted(set(d2) & set(d4))
    missing_m2 = sorted(set(d4) - set(d2))
    missing_m4 = sorted(set(d2) - set(d4))
    y = np.array([d2[k][0] for k in common])
    s2 = np.array([d2[k][1] for k in common])
    s4 = np.array([d4[k][1] for k in common])

    p2 = (s2 >= t2).astype(int)
    p4 = (s4 >= t4).astype(int)

    m2m = prf(y, p2)
    m4m = prf(y, p4)

    agree = float(np.mean(p2 == p4))
    kappa = float(cohen_kappa_score(p2, p4))

    c2 = (p2 == y)
    c4 = (p4 == y)
    both_ok = int(np.sum(c2 & c4))
    both_bad = int(np.sum(~c2 & ~c4))
    only_m2 = int(np.sum(c2 & ~c4))
    only_m4 = int(np.sum(~c2 & c4))

    n_m2_err = int(np.sum(~c2))
    corrected = only_m4
    corrected_rate = corrected / n_m2_err if n_m2_err else 0.0
    n_m2_ok = int(np.sum(c2))
    broken = only_m2
    broken_rate = broken / n_m2_ok if n_m2_ok else 0.0

    or_pred = (p2 | p4).astype(int)
    and_pred = (p2 & p4).astype(int)
    orm = prf(y, or_pred)
    andm = prf(y, and_pred)

    pearson = float(np.corrcoef(s2, s4)[0, 1])
    spearman = float(spearmanr(s2, s4).correlation)

    d2v, d4v = load_scores(m2_dir, "validation"), load_scores(m4_dir, "validation")
    tv2, tv4 = load_threshold(m2_dir), load_threshold(m4_dir)
    kv = sorted(set(d2v) & set(d4v))
    yv = np.array([d2v[k][0] for k in kv])
    v2 = np.array([d2v[k][1] for k in kv])
    v4 = np.array([d4v[k][1] for k in kv])

    lr = LogisticRegression(max_iter=1000)
    lr.fit(np.stack([v2, v4], axis=1), yv)
    val_prob = lr.predict_proba(np.stack([v2, v4], axis=1))[:, 1]
    cands = np.unique(val_prob)
    if cands.size > 512:
        cands = np.quantile(cands, np.linspace(0, 1, 512))
    best_thr, best_f1 = 0.5, -1.0
    for thr in cands:
        f = f1_score(yv, (val_prob >= thr).astype(int))
        if f > best_f1:
            best_thr, best_f1 = float(thr), float(f)
    test_prob = lr.predict_proba(np.stack([s2, s4], axis=1))[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)
    lrm = prf(y, test_pred)
    lrm["auc"] = float(roc_auc_score(y, test_prob))

    report = {
        "models": {
            "M2": {"dir": str(m2_dir), "threshold": t2},
            "M4": {"dir": str(m4_dir), "threshold": t4},
        },
        "split": args.split,
        "pairs": {
            "aligned": len(common),
            "missing_in_m2": len(missing_m2),
            "missing_in_m4": len(missing_m4),
            "positive": int(np.sum(y == 1)),
            "negative": int(np.sum(y == 0)),
        },
        "binary": {
            "agreement": round(agree, 4),
            "kappa": round(kappa, 4),
            "table": {"both_ok": both_ok, "only_m2_ok": only_m2,
                      "only_m4_ok": only_m4, "both_bad": both_bad},
        },
        "per_model": {
            "M2": m2m,
            "M4": m4m,
        },
        "rescue_at_binary_level": {
            "m2_errors": n_m2_err,
            "m4_corrects_m2": corrected,
            "m4_corrects_m2_rate": round(corrected_rate, 4),
            "m2_corrects": n_m2_ok,
            "m4_breaks_m2": broken,
            "m4_breaks_m2_rate": round(broken_rate, 4),
            "net_correctable": corrected - broken,
        },
        "fusion_bounds": {
            "or_rule": orm,
            "and_rule": andm,
        },
        "score_correlation": {
            "pearson": round(pearson, 4),
            "spearman": round(spearman, 4),
        },
        "logistic_fusion": {
            "fit_split": "validation",
            "eval_split": args.split,
            "val_threshold": best_thr,
            "val_f1": round(best_f1, 4),
            "test": lrm,
        },
        "verdict": {
            "best_single_f1": max(m2m["f1"], m4m["f1"]),
            "fusion_test_f1": lrm["f1"],
            "fusion_beats_best_single": bool(lrm["f1"] > max(m2m["f1"], m4m["f1"])),
            "net_correctable": corrected - broken,
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"da luu: {args.output}")


if __name__ == "__main__":
    main()