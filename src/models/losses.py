"""Focal BCE và Supervised Contrastive loss cho huấn luyện M1/M2 (B5/B6)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = 0.5,
) -> torch.Tensor:
    """Focal binary cross entropy trực tiếp trên logits.

    `gamma` giảm trọng số các mẫu dễ (easy negatives chiếm đa số trong BCB),
    `alpha` cân bằng lớp (dataset đã 50:50 nên mặc định 0.5).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    weight = (1.0 - p_t).clamp(min=0.0) ** gamma
    if alpha is not None:
        weight = weight * (targets * alpha + (1.0 - targets) * (1.0 - alpha))
    return (weight * bce).mean()


def supcon_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive trên cặp embedding (ha, hb).

    Với cặp clone (label=1), `ha_i` và `hb_i` là positives của nhau; mọi embedding
    khác trong batch là negatives. Cặp non-clone không có positive (chỉ đóng góp
    làm negative cho các cặp clone) — tránh ép tách các mẫu non-clone không cần thiết.

    Trả về 0 nếu batch không có cặp clone nào.
    """
    labels = labels.reshape(-1).long()
    positives = labels == 1
    if not bool(positives.any()):
        return emb_a.sum() * 0.0

    z = torch.cat([F.normalize(emb_a, dim=1), F.normalize(emb_b, dim=1)], dim=0)
    batch = emb_a.shape[0]
    logits = (z @ z.T) / temperature  # 2B x 2B

    idx = torch.arange(batch, device=z.device)
    # Với clone i: positive là i (a_i) và i+B (b_i); loại self bằng mask.
    pos_mask = torch.zeros(2 * batch, 2 * batch, dtype=torch.bool, device=z.device)
    for i in torch.nonzero(positives, as_tuple=False).reshape(-1):
        pos_mask[i, i + batch] = True
        pos_mask[i + batch, i] = True
    self_mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    pos_mask = pos_mask & ~self_mask

    log_prob = F.log_softmax(logits, dim=1)
    n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
    per_sample = (log_prob * pos_mask.float()).sum(dim=1) / n_pos
    active = torch.cat([positives, positives], dim=0).float()
    return -(per_sample * active).sum() / active.sum().clamp(min=1.0)
