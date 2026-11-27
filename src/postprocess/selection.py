"""Tối ưu chọn ngưỡng/trọng số trên tập validation LỚN + stacking logistic.

Protocol chống overfit:
- Chia validation thành K fold.
- Với mỗi fold: fit isotonic calibration + (tuỳ chọn) logistic stacking trên K-1 fold,
  dự đoán fold giữ lại -> OOF probabilities.
- Chọn ngưỡng F1 và trọng số ensemble bằng OOF (không dùng test).
- Test chỉ đánh giá cuối kèm bootstrap CI + McNemar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from src.postprocess.calibration import apply_calibration, best_f1_threshold, fit_isotonic, sigmoid
from src.postprocess.metrics import bootstrap_f1, compute_metrics, mcnemar


@dataclass
class SelectionResult:
    kind: str
    oof_probs: np.ndarray
    threshold: float
    oof_f1: float
    params: dict[str, Any]


def isotonic_oof(z: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """OOF xác suất đã calibrate bằng isotonic."""
    out = np.empty_like(z, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(z):
        params = fit_isotonic(z[tr], y[tr])
        out[te] = apply_calibration(z[te], "isotonic", params)
    return out


def ensemble_oof(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    step: float = 0.05,
) -> tuple[float, np.ndarray, float]:
    """Quét trọng số w cho A bằng OOF; trả (w, oof_probs, oof_f1)."""
    best: tuple[float, np.ndarray, float] | None = None
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        w = round(float(w), 4)
        mixed = w * probs_a + (1.0 - w) * probs_b
        # OOF F1: mỗi fold tự chọn ngưỡng rồi đánh giá fold giữ lại
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        preds = np.empty_like(mixed, dtype=int)
        for tr, te in kf.split(mixed):
            t, _ = best_f1_threshold(y[tr], mixed[tr])
            preds[te] = (mixed[te] >= t).astype(int)
        f1 = float(2 * np.sum((y == 1) & (preds == 1)) / max(
            2 * np.sum((y == 1) & (preds == 1)) + np.sum((y == 0) & (preds == 1)) + np.sum((y == 1) & (preds == 0)), 1
        ))
        if best is None or f1 > best[2]:
            best = (w, mixed, f1)
    assert best is not None
    return best


def stacking_oof(
    z_a: np.ndarray,
    z_b: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Logistic regression stacking trên logits của 2 model, OOF."""
    out = np.empty_like(y, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = np.column_stack([z_a, z_b])
    for tr, te in kf.split(X):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def threshold_from_oof(y: np.ndarray, oof_probs: np.ndarray) -> tuple[float, float]:
    """Chốt ngưỡng quét F1 trên toàn OOF validation."""
    return best_f1_threshold(y, oof_probs)


def apply_stacking(
    z_a_test: np.ndarray, z_b_test: np.ndarray, z_a_val: np.ndarray, z_b_val: np.ndarray, y_val: np.ndarray
) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(np.column_stack([z_a_val, z_b_val]), y_val)
    return clf.predict_proba(np.column_stack([z_a_test, z_b_test]))[:, 1]


def evaluate_system(
    name: str,
    y_val: np.ndarray,
    oof_val: np.ndarray,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    thr, oof_f1 = threshold_from_oof(y_val, oof_val)
    m = compute_metrics(y_test, test_probs, thr)
    return {
        "name": name,
        "threshold": round(float(thr), 4),
        "oof_f1": round(float(oof_f1), 4),
        "test": m,
        "test_f1_bootstrap": bootstrap_f1(y_test, test_probs, thr, n_boot=n_boot, seed=seed),
        "val_oof_prob": oof_val,
        "test_prob": test_probs,
    }


def compare_significance(
    y_test: np.ndarray,
    sys_a: dict[str, Any],
    sys_b: dict[str, Any],
) -> dict[str, Any]:
    return mcnemar(y_test, sys_a["test_prob"], sys_a["threshold"], sys_b["test_prob"], sys_b["threshold"])
