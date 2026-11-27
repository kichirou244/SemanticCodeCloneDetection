"""Checkpoint averaging (SWA-lite) cho M1/M2 DiffMul.

Chỉ có 2 checkpoint được lưu lại sau mỗi lần train: `best_checkpoint.pt` và
`last_checkpoint.pt`. Module này lấy trung bình trọng số `model_state` của chúng
để tạo một biến thể SWA, rồi để pipeline quyết định có dùng biến thể đó hay không
dựa trên validation (không bao giờ dựa trên test).

Nếu 2 checkpoint cùng epoch (trường hợp M1: best ở epoch 8 = epoch cuối), hàm
trả về None để pipeline bỏ qua — tránh inference lãng phí trên weights trùng nhau.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]], weights: list[float] | None = None) -> dict[str, torch.Tensor]:
    """Lấy trung bình có trọng số của nhiều state_dict.

    Tensor floating-point được trung bình; buffer không phải floating-point
    (vd. `num_batches_tracked`) lấy từ state_dict đầu tiên.
    """
    if not state_dicts:
        raise ValueError("Danh sách state_dict rỗng")
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError("Số weights phải bằng số state_dict")
    total = float(sum(weights))
    weights = [w / total for w in weights]

    keys = set(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if set(sd.keys()) != keys:
            raise ValueError("Các state_dict có bộ khóa không khớp nhau")

    averaged: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        tensors = [sd[key] for sd in state_dicts]
        if all(torch.is_floating_point(t) for t in tensors):
            acc = tensors[0].float() * weights[0]
            for t, w in zip(tensors[1:], weights[1:]):
                acc = acc + t.float() * w
            averaged[key] = acc.to(dtype=tensors[0].dtype)
        else:
            averaged[key] = tensors[0].clone()
    return averaged


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_swa_checkpoint(
    best_path: str | Path,
    last_path: str | Path,
    out_path: str | Path | None = None,
    alpha: float = 0.5,
) -> dict[str, Any] | None:
    """Trộn best + last. Trả về dict checkpoint (chưa chắc đã ghi file) hoặc None.

    `alpha` là trọng số của `best`; trọng số của `last` là `1 - alpha`.
    """
    best = load_checkpoint(best_path)
    last = load_checkpoint(last_path)

    epoch_best = int(best.get("epoch", -1))
    epoch_last = int(last.get("epoch", -1))
    if epoch_best == epoch_last:
        return None

    avg_state = average_state_dicts([best["model_state"], last["model_state"]], [alpha, 1.0 - alpha])
    out: dict[str, Any] = dict(best)
    out["model_state"] = avg_state
    out["swa_info"] = {
        "sources": [str(Path(best_path).name), str(Path(last_path).name)],
        "epochs": [epoch_best, epoch_last],
        "alpha_best": alpha,
        "note": "SWA-lite: trung bình best + last. Ngưỡng kế thừa từ best checkpoint.",
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, out_path)
    return out


def average_checkpoint_paths(
    paths: list[str | Path],
    out_path: str | Path,
    weights: list[float] | None = None,
) -> Path:
    """Trung bình nhiều checkpoint (dùng cho SWA snapshots trong tương lai)."""
    if not paths:
        raise ValueError("Không có checkpoint để trung bình")
    ckpts = [load_checkpoint(p) for p in paths]
    avg_state = average_state_dicts([c["model_state"] for c in ckpts], weights)
    out: dict[str, Any] = dict(ckpts[0])
    out["model_state"] = avg_state
    out["swa_info"] = {
        "sources": [str(Path(p).name) for p in paths],
        "epochs": [int(c.get("epoch", -1)) for c in ckpts],
        "weights": weights,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return out_path
