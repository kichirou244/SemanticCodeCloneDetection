"""Kiến trúc Đồ thị Đa Quan hệ (Heterogeneous Relational CPG-GNN) cho M4 và M5-Pro.

Đặc tính nâng cấp so với TypedGATLayer (DEC-033):
1. Phân tách rõ ràng không gian biểu diễn của 6 loại quan hệ đồ thị CPG:
   - AST: Cấu trúc ngữ pháp phân cấp (cây cú pháp trừu tượng).
   - CFG: Luồng điều khiển thực thi (rẽ nhánh điều kiện, vòng lặp, tuần tự).
   - ARGUMENT / PARAMETER_OUT: Luồng truyền nhận tham số và giá trị dữ liệu.
   - TARG / EVAL_ORDER: Tham số kiểu dữ liệu và thứ tự tính toán biểu thức.
2. Mỗi quan hệ r sở hữu ma trận biến đổi W_r riêng biệt kèm chuẩn hóa bậc (degree normalization).
3. Triển khai bằng PyTorch thuần (vectorized index_add) để đảm bảo 100% tương thích,
   không phụ thuộc vào binary C++ PyG compiled trên bất kỳ môi trường Kaggle/Cloud nào.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool

from src.representations.cpg.schema import FEATURE_DIM, N_EDGE_TYPES, N_NODE_TYPES


class HeteroRelationalConv(nn.Module):
    """Lớp tích chập đồ thị đa quan hệ với ma trận biến đổi độc lập theo từng loại cạnh."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int = N_EDGE_TYPES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations

        # W_r cho từng quan hệ đồ thị [num_relations, in_dim, out_dim]
        self.relation_weights = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        # W_self cho đỉnh tự thân [in_dim, out_dim]
        self.self_weight = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Trọng số chú ý quan hệ (Relation Importance Attention)
        self.relation_attn = nn.Parameter(torch.ones(num_relations))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.zeros_(self.bias)
        nn.init.constant_(self.relation_attn, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Đặc trưng đỉnh [N, in_dim]
            edge_index: Cặp đỉnh nối [2, E]
            edge_type: Nhãn loại cạnh [E] (0..num_relations-1)
        """
        N = x.size(0)
        out = self.self_weight(x)
        src = edge_index[0]
        dst = edge_index[1]

        # Chuẩn hóa trọng số quan hệ
        rel_weights = F.softmax(self.relation_attn, dim=0)

        # Lan truyền thông điệp riêng biệt cho từng loại quan hệ
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if not mask.any():
                continue

            r_src = src[mask]
            r_dst = dst[mask]
            W_r = self.relation_weights[r]  # [in_dim, out_dim]

            # Biến đổi đặc trưng đỉnh nguồn theo ma trận quan hệ r
            msg = torch.matmul(x[r_src], W_r)  # [E_r, out_dim]

            # Chuẩn hóa bậc tới của đỉnh đích (in-degree normalization)
            deg = torch.zeros(N, device=x.device, dtype=x.dtype)
            deg.index_add_(0, r_dst, torch.ones_like(r_dst, dtype=x.dtype))
            deg = deg.clamp(min=1.0).unsqueeze(-1)
            norm_factor = 1.0 / deg[r_dst]

            # Gom thông điệp và tích hợp trọng số quan hệ
            weighted_msg = msg * norm_factor * rel_weights[r]
            out = out.index_add(0, r_dst, weighted_msg)

        out = out + self.bias
        return self.dropout(self.norm(F.gelu(out)))


class HeteroCPGEncoder(nn.Module):
    """Bộ mã hóa CPG đa quan hệ kết hợp cấu trúc cú pháp AST, luồng điều khiển CFG và dữ liệu."""

    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 3,
        num_relations: int = N_EDGE_TYPES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.node_emb = nn.Embedding(N_NODE_TYPES, hidden)
        self.feat_lin = nn.Linear(FEATURE_DIM, hidden)
        self.in_norm = nn.LayerNorm(hidden)

        self.layers = nn.ModuleList([
            HeteroRelationalConv(hidden, hidden, num_relations=num_relations, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Cơ chế pooling phân cấp đa tầng (Hierarchical Graph Pooling)
        self.out_norm = nn.LayerNorm(2 * hidden)

    def forward(self, batch) -> torch.Tensor:
        # Nhúng loại đỉnh và đặc trưng đối số
        x = self.in_norm(self.node_emb(batch.node_type) + self.feat_lin(batch.x))

        # Lan truyền qua các tầng Heterogeneous Relational Conv kèm kết nối tàn dư (Residual)
        for layer in self.layers:
            x = x + layer(x, batch.edge_index, batch.edge_type)

        # Gom tụ đồ thị toàn cục (Global Mean + Max Pooling)
        h_mean = global_mean_pool(x, batch.batch)
        h_max = global_max_pool(x, batch.batch)
        h_graph = torch.cat([h_mean, h_max], dim=-1)
        return self.out_norm(h_graph)


class HeteroCPGGNNPairClassifier(nn.Module):
    """Mô hình phân loại cặp hàm dựa trên Heterogeneous CPG-GNN."""

    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 3,
        num_relations: int = N_EDGE_TYPES,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = HeteroCPGEncoder(
            hidden=hidden,
            num_layers=num_layers,
            num_relations=num_relations,
            dropout=dropout,
        )
        self.enc_dim = 2 * hidden
        self.dropout = nn.Dropout(dropout)

        # Tương tác cặp: [h_a, h_b, |h_a - h_b|, h_a * h_b] -> 4 * enc_dim
        self.head = nn.Sequential(
            nn.Linear(4 * self.enc_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def encode_side(self, batch) -> torch.Tensor:
        return self.encoder(batch)

    def forward(self, batch_a, batch_b) -> torch.Tensor:
        ha = self.encode_side(batch_a)
        hb = self.encode_side(batch_b)
        diff = torch.abs(ha - hb)
        prod = ha * hb
        pair = torch.cat([ha, hb, diff, prod], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
