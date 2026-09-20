"""Metrics classification + retrieval (không trộn lẫn hai nhóm)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def recall_at_k(ranked: list[list[int]], k: int) -> float:
    """ranked: mỗi query là list relevance (1/0) theo thứ hạng."""
    hits = sum(1 for r in ranked if any(r[:k]))
    return hits / max(1, len(ranked))


def mrr(ranked: list[list[int]]) -> float:
    scores = []
    for r in ranked:
        try:
            scores.append(1.0 / (r.index(1) + 1))
        except ValueError:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0
