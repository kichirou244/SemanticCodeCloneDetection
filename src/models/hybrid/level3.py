"""Mô hình Lai Ghép Cấp độ 3 (Level 3 End-to-End Deep Graph-Transformer).

Kiến trúc liên kết ngữ nghĩa sâu đa phương thức (Cross-Modal Co-Adaptation):
  1. Text Backbone: GraphCodeBERT (125M params) mã hóa chuỗi mã nguồn và luồng dữ liệu thô.
  2. Graph Backbone: CPGEncoder (3 lớp TypedGATLayer có nhúng cạnh 32 chiều) mã hóa cấu trúc AST + CFG + CDG + DDG.
  3. Cổng điều biến 2 chiều (Bidirectional Dynamic Gated Cross-Modal Fusion):
     - Text-guided Graph Gate: Dùng ngữ cảnh Transformer để lọc nhiễu cú pháp từ CPG.
     - Graph-guided Text Gate: Dùng cấu trúc đồ thị điều biến các token ranh giới logic.
  4. Bộ so sánh Siamese Difference Head (DEC-016) chuẩn hóa LayerNorm + GELU.
  5. Cơ chế Discriminative Learning Rates: Học đồng thời cả Transformer và GAT với tốc độ học riêng biệt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.cpg_gnn.model import CPGEncoder

logger = logging.getLogger("hybrid_level3")


class BidirectionalCrossModalFusion(nn.Module):
    """Cơ chế Cổng Điều biến Đa phương thức Hai Chiều (Bidirectional Cross-Modal Gating)."""

    def __init__(self, text_dim: int = 768, graph_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.graph_dim = graph_dim
        total_dim = text_dim + graph_dim

        # Cổng điều biến đồ thị từ đặc trưng văn bản & đồ thị
        self.graph_gate = nn.Sequential(
            nn.Linear(total_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

        # Cổng điều biến văn bản từ đặc trưng cấu trúc & văn bản
        self.text_gate = nn.Sequential(
            nn.Linear(total_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(total_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_text: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_text, h_graph], dim=-1)
        g_graph = self.graph_gate(combined)
        g_text = self.text_gate(combined)

        h_text_mod = h_text * g_text
        h_graph_mod = h_graph * g_graph

        fused = torch.cat([h_text_mod, h_graph_mod], dim=-1)
        return self.dropout(self.out_norm(fused))


class DeepGraphTransformerPairClassifier(nn.Module):
    """M5 Level 3: Phân loại cặp hàm lai ghép End-to-End giữa GraphCodeBERT và CPG-GNN."""

    def __init__(
        self,
        hf_id: str = "microsoft/graphcodebert-base",
        graph_hidden: int = 128,
        graph_num_layers: int = 3,
        graph_heads: int = 4,
        graph_dropout: float = 0.2,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
        freeze_text: bool = False,
        freeze_graph: bool = False,
    ) -> None:
        super().__init__()
        self.hf_id = hf_id
        self.text_dim = 768
        self.graph_dim = 2 * graph_hidden  # mean_pool (128) + max_pool (128) = 256
        self.fused_dim = self.text_dim + self.graph_dim  # 1024

        self.freeze_text = freeze_text
        self.freeze_graph = freeze_graph

        # 1. Encoders thành phần
        self.text_encoder = AutoModel.from_pretrained(hf_id)
        self.graph_encoder = CPGEncoder(
            hidden=graph_hidden,
            num_layers=graph_num_layers,
            heads=graph_heads,
            dropout=graph_dropout,
        )

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_graph:
            for p in self.graph_encoder.parameters():
                p.requires_grad = False

        # 2. Cổng tương tác lai ghép đa phương thức
        self.fusion = BidirectionalCrossModalFusion(
            text_dim=self.text_dim,
            graph_dim=self.graph_dim,
            dropout=head_dropout,
        )

        # 3. Phân loại cặp Siamese Difference Head (DEC-016)
        pair_in_dim = 3 * self.fused_dim  # 3 * 1024 = 3072
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(pair_in_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 4),
            nn.LayerNorm(head_hidden // 4),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 4, 1),
        )

    def encode_snippet(self, text_inputs: dict[str, torch.Tensor], graph_batch: Any) -> torch.Tensor:
        out = self.text_encoder(**text_inputs)
        h_text = out.last_hidden_state[:, 0, :]  # (B, 768)
        h_graph = self.graph_encoder(graph_batch)  # (B, 256)
        return self.fusion(h_text, h_graph)  # (B, 1024)

    def forward(
        self,
        text_a: dict[str, torch.Tensor],
        graph_a: Any,
        text_b: dict[str, torch.Tensor],
        graph_b: Any,
    ) -> torch.Tensor:
        h_a = self.encode_snippet(text_a, graph_a)
        h_b = self.encode_snippet(text_b, graph_b)
        pair_diff = torch.abs(h_a - h_b)
        pair_rep = torch.cat([h_a, h_b, pair_diff], dim=-1)
        return self.head(pair_rep).squeeze(-1)

    def get_optimizer_param_groups(
        self,
        lr_text: float = 1.5e-5,
        lr_graph: float = 1e-4,
        lr_head: float = 2e-4,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Nhóm tham số cho optimizer với discriminative learning rates."""
        groups = []
        if not self.freeze_text:
            groups.append({
                "params": [p for p in self.text_encoder.parameters() if p.requires_grad],
                "lr": lr_text,
                "weight_decay": weight_decay,
            })
        if not self.freeze_graph:
            groups.append({
                "params": [p for p in self.graph_encoder.parameters() if p.requires_grad],
                "lr": lr_graph,
                "weight_decay": 0.001,
            })
        head_params = list(self.fusion.parameters()) + list(self.head.parameters())
        groups.append({
            "params": [p for p in head_params if p.requires_grad],
            "lr": lr_head,
            "weight_decay": weight_decay,
        })
        return groups

    def load_warm_start(
        self,
        m2_ckpt_path: str | Path | None = None,
        m4_ckpt_path: str | Path | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> dict[str, bool]:
        """Tải trọng số khởi tạo ấm từ các mô hình đã tinh chỉnh tối ưu (M2-DR và M4-EP)."""
        status = {"text": False, "graph": False}
        if m2_ckpt_path:
            p2 = Path(m2_ckpt_path)
            if p2.exists():
                try:
                    ckpt2 = torch.load(p2, map_location=device, weights_only=False)
                    state2 = ckpt2.get("model_state", ckpt2)
                    enc_state2 = {k.replace("encoder.", "", 1): v for k, v in state2.items() if k.startswith("encoder.")}
                    self.text_encoder.load_state_dict(enc_state2, strict=False)
                    logger.info("Đã nạp warm-start GraphCodeBERT từ %s (%d keys)", p2.name, len(enc_state2))
                    status["text"] = True
                except Exception as e:
                    logger.warning("Không thể nạp warm-start M2: %s", e)
        if m4_ckpt_path:
            p4 = Path(m4_ckpt_path)
            if p4.exists():
                try:
                    ckpt4 = torch.load(p4, map_location=device, weights_only=False)
                    state4 = ckpt4.get("model_state", ckpt4)
                    enc_state4 = {k.replace("encoder.", "", 1): v for k, v in state4.items() if k.startswith("encoder.")}
                    self.graph_encoder.load_state_dict(enc_state4, strict=False)
                    logger.info("Đã nạp warm-start CPG-GNN từ %s (%d keys)", p4.name, len(enc_state4))
                    status["graph"] = True
                except Exception as e:
                    logger.warning("Không thể nạp warm-start M4: %s", e)
        return status

    def param_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in self.text_encoder.parameters())
        graph_total = sum(p.numel() for p in self.graph_encoder.parameters())
        head_total = sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.fusion.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "text": text_total,
            "graph": graph_total,
            "fusion_head": head_total,
        }
