"""Module Tương Tác Chú Ý Đa Phương Thức (Cross-Modal Token-to-Node Attention) & Căn Chỉnh Tương Phản (Contrastive Alignment).

Đặc tính kiến trúc SOTA (DEC-034):
1. Token-to-Node Cross-Attention:
   - Cho phép từng mã lệnh (Token từ GraphCodeBERT Layer 12) truy vấn trực tiếp vào tập đỉnh đồ thị CPG
     (nút hàm, biến số, cấu trúc điều kiện, lệnh gọi).
   - Tính toán ma trận Attention liên phương thức giữa chuỗi tuần tự và cấu trúc đồ thị.
2. CodeCLIP / InfoNCE Contrastive Alignment:
   - Ép không gian biểu diễn của văn bản mã nguồn (Text Stream) và đồ thị thuộc tính mã nguồn (CPG Stream)
     đồng nhất với nhau trước khi thực hiện phân loại clone.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenNodeCrossAttention(nn.Module):
    """Lớp Cross-Attention giữa chuỗi token văn bản và tập đỉnh đồ thị CPG."""

    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        attn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        self.proj_q = nn.Linear(text_dim, attn_dim)
        self.proj_k = nn.Linear(graph_dim, attn_dim)
        self.proj_v = nn.Linear(graph_dim, attn_dim)

        self.out_proj = nn.Linear(attn_dim, attn_dim)
        self.norm = nn.LayerNorm(attn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_tokens: torch.Tensor,
        graph_nodes: torch.Tensor,
        graph_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            text_tokens: [B, L, text_dim]
            graph_nodes: [B, N, graph_dim]
            graph_mask: [B, N] bool tensor (True là hợp lệ, False là padding)
        Returns:
            Vector biểu diễn chuỗi đã được làm giàu cấu trúc đồ thị [B, attn_dim]
        """
        B, L, _ = text_tokens.shape
        _, N, _ = graph_nodes.shape

        # Chiếu vào không gian attention
        Q = self.proj_q(text_tokens).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D]
        K = self.proj_k(graph_nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = self.proj_v(graph_nodes).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L, N]

        if graph_mask is not None:
            # mask: [B, 1, 1, N]
            mask = graph_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Gom tụ giá trị
        context = torch.matmul(attn_weights, V)  # [B, H, L, D]
        context = context.transpose(1, 2).contiguous().view(B, L, self.attn_dim)
        context = self.out_proj(context)

        # Pooling trên chuỗi token: Mean pool token đại diện hàm
        pooled = context.mean(dim=1)  # [B, attn_dim]
        return self.norm(pooled)


class InfoNCEContrastiveAlignmentLoss(nn.Module):
    """Hàm mất mát căn chỉnh tương phản đa phương thức InfoNCE (CodeCLIP)."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """Tính toán loss căn chỉnh giữa Text và Graph.
        
        Args:
            text_emb: [B, D]
            graph_emb: [B, D]
        """
        # Chuẩn hóa L2
        u = F.normalize(text_emb, dim=-1)
        v = F.normalize(graph_emb, dim=-1)

        # Ma trận tương đồng cosine [B, B]
        sim = torch.matmul(u, v.transpose(0, 1)) / self.temperature

        # Nhãn đối sánh 1-1 trên đường chéo
        targets = torch.arange(sim.size(0), device=sim.device)

        loss_t2g = F.cross_entropy(sim, targets)
        loss_g2t = F.cross_entropy(sim.t(), targets)
        return 0.5 * (loss_t2g + loss_g2t)
