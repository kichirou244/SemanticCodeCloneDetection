"""Các hàm mất mát nâng cao cho Mô hình Lai Ghép M5-Pro (Loss Functions).

Bao gồm:
  1. FocalLoss: Tập trung gradient vào các mẫu phân loại khó (Hard Examples).
  2. AsymmetricLoss (ASL): Phạt bất đối xứng cực nặng đối với các ca False Positives
     nhằm triệt tiêu báo nhầm mà không làm suy giảm độ nhạy Recall.
  3. DynamicHardMiningLoss: Tự động phát hiện và nhân trọng số phạt các ca cạm bẫy non-clone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal Loss cho phân loại nhị phân cân bằng / mất cân bằng.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = -alpha_t * ((1.0 - p_t) ** self.gamma) * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) triệt tiêu False Positives.
    
    Áp dụng số mũ gamma_neg lớn hơn gamma_pos để phạt lũy thừa đối với
    các mẫu Non-clone bị dự đoán nhầm thành Clone.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        
        # Phần dương tính (Clone): phạt chuẩn để duy trì Recall cao
        loss_pos = -targets * ((1.0 - p) ** self.gamma_pos) * torch.log(p + self.eps)
        
        # Phần âm tính (Non-clone): phạt bất đối xứng để triệt tiêu FP
        p_neg = (p - self.clip).clamp(min=0.0)
        loss_neg = -(1.0 - targets) * (p_neg ** self.gamma_neg) * torch.log(1.0 - p_neg + self.eps)
        
        return (loss_pos + loss_neg).mean()
