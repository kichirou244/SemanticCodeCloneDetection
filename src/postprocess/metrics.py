"""Metrics nhị phân, bootstrap CI và McNemar cho so sánh paired trên test."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Tính metrics tại một ngưỡng cố định. `y_prob` có thể là score đã calibrate."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    return {
        "threshold": round(float(threshold), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(auc, 4),
    }


def bootstrap_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap percentile CI cho F1 tại ngưỡng cố định."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            scores[i] = np.nan
            continue
        scores[i] = f1_score(yt, (yp >= threshold).astype(int), zero_division=0)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return {"mean": None, "sd": None, "ci95": [None, None], "n_boot": n_boot}
    return {
        "mean": round(float(scores.mean()), 4),
        "sd": round(float(scores.std()), 4),
        "ci95": [round(float(np.quantile(scores, 0.025)), 4), round(float(np.quantile(scores, 0.975)), 4)],
        "n_boot": n_boot,
    }


def mcnemar(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    thr_a: float,
    prob_b: np.ndarray,
    thr_b: float,
) -> dict[str, Any]:
    """McNemar exact binomial giữa hai hệ thống trên cùng một tập test."""
    y_true = np.asarray(y_true, dtype=int)
    pred_a = (np.asarray(prob_a) >= thr_a).astype(int)
    pred_b = (np.asarray(prob_b) >= thr_b).astype(int)
    correct_a, correct_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(correct_a & ~correct_b))  # A đúng, B sai
    c = int(np.sum(~correct_a & correct_b))  # A sai, B đúng
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant": b + c,
        "p_value": p,
        "significant": bool(p < 0.05),
        "note": "p < 0.05 => hai hệ thống sai khác nhau có ý nghĩa (A = baseline, B = candidate).",
    }
