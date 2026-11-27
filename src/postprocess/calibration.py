"""Calibration xác suất và chọn ngưỡng F1 ổn định bằng OOF.

Hiện tại M1/M2 DiffMul dùng score `sigmoid(logits)` chưa calibrate và chọn
ngưỡng bằng quét F1 trên một tập validation 3k. Hệ quả: ngưỡng tối ưu nhảy
0.1 -> 0.82 giữa các epoch dù F1 gần như không đổi. Module này:

1. Fit temperature scaling / Platt scaling / isotonic regression trên logits validation.
2. Chọn phương pháp có NLL validation thấp nhất (proper scoring rule).
3. Chọn ngưỡng bằng 5-fold OOF trên validation, báo cả F1 OOF (ước tính vô hướng
   của quy trình chọn ngưỡng) lẫn ngưỡng chốt trên toàn validation.

Không dùng test cho bất kỳ bước fit nào.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.postprocess.metrics import compute_metrics

Method = Literal["none", "temperature", "platt", "isotonic"]
EPS = 1e-7


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def log_sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return -np.logaddexp(0.0, -z)


def nll_from_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood trực tiếp trên logits (ổn định số học)."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    log_p1 = log_sigmoid(z)
    log_p0 = log_sigmoid(-z)
    return float(-np.mean(y * log_p1 + (1.0 - y) * log_p0))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Tìm T > 0 sao cho NLL(sigmoid(z / T), y) nhỏ nhất."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    def objective(log_t: float) -> float:
        return nll_from_logits(z / np.exp(log_t), y)

    res = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    t = float(np.exp(res.x))
    return {"T": round(t, 6)}


def fit_platt(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Platt scaling: p = sigmoid(a * z + b), fit bằng L-BFGS trên NLL."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)

    def objective(ab: np.ndarray) -> tuple[float, np.ndarray]:
        logits = ab[0] * z + ab[1]
        val = nll_from_logits(logits, y)
        # gradient: (sigmoid(logits) - y) * [z, 1]
        g = sigmoid(logits) - y
        grad = np.array([float(np.mean(g * z)), float(np.mean(g))])
        return val, grad

    res = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
    a, b = (float(v) for v in res.x)
    return {"a": round(a, 6), "b": round(b, 6)}


def fit_isotonic(z: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Isotonic regression trên sigmoid(logits). Lưu breakpoints để serialize."""
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(sigmoid(z), np.asarray(y, dtype=int))
    return {
        "X_thresholds_": [float(v) for v in ir.X_thresholds_],
        "y_thresholds_": [float(v) for v in ir.y_thresholds_],
    }


def apply_calibration(z: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if method == "none":
        p = sigmoid(z)
    elif method == "temperature":
        p = sigmoid(z / float(params["T"]))
    elif method == "platt":
        p = sigmoid(float(params["a"]) * z + float(params["b"]))
    elif method == "isotonic":
        # Tương đương IsotonicRegression(out_of_bounds="clip").predict(): np.interp
        # tự clip ở hai đầu, không cần tái tạo object sklearn từ breakpoints.
        p = np.interp(
            sigmoid(z),
            np.asarray(params["X_thresholds_"], dtype=float),
            np.asarray(params["y_thresholds_"], dtype=float),
        )
    else:
        raise ValueError(f"Phương pháp calibration không hợp lệ: {method}")
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """ECE với các bin độ rộng bằng nhau trên [0, 1]."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return round(ece, 6)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return round(float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2)), 6)


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Quét ngưỡng tối ưu F1 (vectorized)."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.arange(0.01, 1.0, 0.01)
    preds = p[None, :] >= grid[:, None]
    tp = np.sum(preds & (y[None, :] == 1), axis=1)
    fp = np.sum(preds & (y[None, :] == 0), axis=1)
    fn = np.sum((~preds) & (y[None, :] == 1), axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    i = int(np.argmax(f1))
    return float(grid[i]), float(f1[i])


@dataclass
class OOFThresholdResult:
    """Kết quả chọn ngưỡng bằng OOF trên validation."""

    threshold: float
    oof_f1: float
    oof_metrics: dict[str, Any]
    fold_thresholds: list[float] = field(default_factory=list)
    fold_f1: list[float] = field(default_factory=list)


def select_threshold_oof(
    y: np.ndarray,
    p: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    grid: np.ndarray | None = None,
) -> OOFThresholdResult:
    """Chọn ngưỡng ở từng fold rồi đánh giá trên fold giữ lại.

    - `threshold`: ngưỡng chốt, quét F1 trên toàn validation (dùng cho test).
    - `oof_f1`: F1 khi mỗi fold dùng ngưỡng được fit từ 4 fold còn lại — ước tính
      vô hướng cho độ ổn định của quy trình chọn ngưỡng.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if grid is None:
        grid = np.arange(0.01, 1.0, 0.01)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_prob = np.empty_like(p)
    oof_threshold_per_sample = np.empty_like(p)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(p)):
        t_fold, _ = best_f1_threshold(y[tr_idx], p[tr_idx], grid)
        oof_prob[te_idx] = p[te_idx]
        oof_threshold_per_sample[te_idx] = t_fold

    oof_pred = (oof_prob >= oof_threshold_per_sample).astype(int)
    oof_metrics = compute_metrics(y, oof_prob, 0.5)
    # compute_metrics nhận 1 ngưỡng cố định; tính lại F1 OOF theo từng sample
    tp = int(np.sum((y == 1) & (oof_pred == 1)))
    tn = int(np.sum((y == 0) & (oof_pred == 0)))
    fp = int(np.sum((y == 0) & (oof_pred == 1)))
    fn = int(np.sum((y == 1) & (oof_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    oof_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    oof_metrics.update(
        {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(oof_f1, 4),
            "accuracy": round(float((oof_pred == y).mean()), 4),
        }
    )

    threshold, _ = best_f1_threshold(y, p, grid)
    fold_thresholds = []
    fold_f1 = []
    for tr_idx, te_idx in kf.split(p):
        t_fold, _ = best_f1_threshold(y[tr_idx], p[tr_idx], grid)
        fold_thresholds.append(round(t_fold, 4))
        fold_f1.append(round(float(((p[te_idx] >= t_fold).astype(int) == y[te_idx]).mean()), 4))

    return OOFThresholdResult(
        threshold=round(threshold, 4),
        oof_f1=round(oof_f1, 4),
        oof_metrics=oof_metrics,
        fold_thresholds=fold_thresholds,
        fold_f1=fold_f1,
    )


@dataclass
class CalibrationFit:
    """Kết quả fit một phương pháp calibration trên validation."""

    method: str
    params: dict[str, Any]
    val_probs: np.ndarray
    nll: float
    ece: float
    brier: float
    threshold_result: OOFThresholdResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "nll": round(self.nll, 6),
            "ece": self.ece,
            "brier": self.brier,
            "threshold": self.threshold_result.threshold,
            "oof_f1": self.threshold_result.oof_f1,
            "oof_metrics": self.threshold_result.oof_metrics,
            "fold_thresholds": self.threshold_result.fold_thresholds,
            "fold_accuracy": self.threshold_result.fold_f1,
        }


def fit_calibration(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    methods: tuple[str, ...] = ("none", "temperature", "platt", "isotonic"),
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit tất cả phương pháp, chọn method có NLL thấp nhất."""
    z = np.asarray(val_logits, dtype=float)
    y = np.asarray(val_labels, dtype=int)

    fits: dict[str, CalibrationFit] = {}
    for method in methods:
        if method == "none":
            params: dict[str, Any] = {}
        elif method == "temperature":
            params = fit_temperature(z, y)
        elif method == "platt":
            params = fit_platt(z, y)
        elif method == "isotonic":
            params = fit_isotonic(z, y)
        else:
            raise ValueError(f"Không hỗ trợ method={method}")

        p = apply_calibration(z, method, params)
        p_for_nll = np.clip(p, EPS, 1.0 - EPS)
        nll = float(-np.mean(y * np.log(p_for_nll) + (1 - y) * np.log(1 - p_for_nll)))
        fits[method] = CalibrationFit(
            method=method,
            params=params,
            val_probs=p,
            nll=round(nll, 6),
            ece=expected_calibration_error(y, p),
            brier=brier_score(y, p),
            threshold_result=select_threshold_oof(y, p, n_folds=n_folds, seed=seed),
        )

    # Chọn theo NLL; tie-break theo ECE rồi theo OOF F1
    chosen = min(fits.values(), key=lambda f: (f.nll, f.ece, -f.threshold_result.oof_f1))
    return {
        "chosen_method": chosen.method,
        "chosen_params": chosen.params,
        "chosen_val_probs": chosen.val_probs,
        "chosen_threshold": chosen.threshold_result.threshold,
        "chosen_oof_f1": chosen.threshold_result.oof_f1,
        "fits": {name: fit.to_dict() for name, fit in fits.items()},
        "fit_objects": fits,
    }
