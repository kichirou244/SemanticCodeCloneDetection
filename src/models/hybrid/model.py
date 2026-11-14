"""Mô hình Lai Đặc trưng Cấp độ 2 (Feature-Level Soft Fusion) M5.

Kết hợp không gian vector biểu diễn ngữ nghĩa chuỗi mã nguồn từ GraphCodeBERT (768 chiều)
và không gian vector biểu diễn cấu trúc đồ thị từ CPG-GNN (256 chiều).
Hỗ trợ cả Concat Fusion cơ bản và Gated Cross-Modal Dynamic Modulation.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.cpg_gnn.model import CPGEncoder


class GatedCrossModalFusion(nn.Module):
    """Cơ chế Cổng Thích ứng Đặc trưng (Dynamic Gated Feature Modulation).

    Đầu vào:
      h_text: (B, text_dim = 768)
      h_graph: (B, graph_dim = 256)
    Cổng g:
      g = sigmoid(Linear([h_text, h_graph] -> graph_dim))
    Đặc trưng cấu trúc sau điều biến:
      h_graph_mod = g * h_graph
    Đặc trưng hợp nhất:
      h_fused = [h_text, h_graph_mod]  # (B, text_dim + graph_dim = 1024)
    """

    def __init__(self, text_dim: int = 768, graph_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(text_dim + graph_dim, graph_dim),
            nn.LayerNorm(graph_dim),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

    def forward(self, h_text: torch.Tensor, h_graph: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_text, h_graph], dim=-1)
        gate = self.gate_fc(combined)
        h_graph_mod = gate * h_graph
        return torch.cat([h_text, h_graph_mod], dim=-1)


class HybridPairClassifier(nn.Module):
    """M5: Phân loại cặp hàm lai ghép GraphCodeBERT + CPG-GNN (Feature-Level Fusion)."""

    def __init__(
        self,
        hf_id: str = "microsoft/graphcodebert-base",
        graph_hidden: int = 128,
        graph_num_layers: int = 3,
        graph_heads: int = 4,
        graph_dropout: float = 0.2,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
        text_dim: int = 768,
        use_gating: bool = True,
        freeze_encoders: bool = True,
        text_encoder: nn.Module | None = None,
        graph_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.hf_id = hf_id
        self.text_dim = text_dim
        self.graph_dim = 2 * graph_hidden  # mean_pool (128) + max_pool (128) = 256
        self.use_gating = use_gating
        self.freeze_encoders = freeze_encoders

        # 1. Encoders thành phần
        self.text_encoder = text_encoder if text_encoder is not None else AutoModel.from_pretrained(hf_id)
        self.graph_encoder = graph_encoder if graph_encoder is not None else CPGEncoder(
            hidden=graph_hidden, num_layers=graph_num_layers, heads=graph_heads, dropout=graph_dropout
        )

        if self.freeze_encoders:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.graph_encoder.parameters():
                p.requires_grad = False

        # 2. Bộ điều biến tương tác đa phương thức
        if self.use_gating:
            self.fusion_layer = GatedCrossModalFusion(self.text_dim, self.graph_dim, dropout=head_dropout)
        else:
            self.fusion_layer = None

        self.fused_dim = self.text_dim + self.graph_dim  # 768 + 256 = 1024

        # 3. Phân loại cặp (Siamese difference head theo chuẩn DEC-016)
        # [h_a ; h_b ; |h_a - h_b|] -> 3 * 1024 = 3072 chiều
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * self.fused_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.Tanh(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.LayerNorm(head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.text_encoder.eval()
            self.graph_encoder.eval()
        return self

    def encode_side(self, text_inputs: dict, graph_batch) -> torch.Tensor:
        out = self.text_encoder(**text_inputs)
        h_text = out.last_hidden_state[:, 0, :]
        h_graph = self.graph_encoder(graph_batch)
        if self.use_gating and self.fusion_layer is not None:
            return self.fusion_layer(h_text, h_graph)
        return torch.cat([h_text, h_graph], dim=-1)

    def forward_from_embeddings(self, h_text_a: torch.Tensor, h_graph_a: torch.Tensor,
                                h_text_b: torch.Tensor, h_graph_b: torch.Tensor) -> torch.Tensor:
        """Forward trực tiếp từ embeddings đã trích xuất sẵn (tăng tốc độ huấn luyện 100x)."""
        if self.use_gating and self.fusion_layer is not None:
            ha = self.fusion_layer(h_text_a, h_graph_a)
            hb = self.fusion_layer(h_text_b, h_graph_b)
        else:
            ha = torch.cat([h_text_a, h_graph_a], dim=-1)
            hb = torch.cat([h_text_b, h_graph_b], dim=-1)

        pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def forward(self, text_a: dict, graph_a, text_b: dict, graph_b) -> torch.Tensor:
        ha = self.encode_side(text_a, graph_a)
        hb = self.encode_side(text_b, graph_b)
        pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def warm_start_text(text_encoder: nn.Module, ckpt_path: str | Path, device: torch.device) -> bool:
    """Nạp trọng số GraphCodeBERT từ checkpoint M2."""
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[warm_start_text] Không tìm thấy {p}")
        return False
    ckpt = torch.load(p, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    encoder_keys = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = text_encoder.load_state_dict(encoder_keys, strict=False)
    print(f"[warm_start_text] Đã nạp thành công từ {p.name} (layers={len(encoder_keys)}, missing={len(missing)})")
    return True


def warm_start_graph(graph_encoder: nn.Module, ckpt_path: str | Path, device: torch.device) -> bool:
    """Nạp trọng số CPG-GNN từ checkpoint M4."""
    p = Path(ckpt_path)
    if not p.exists():
        print(f"[warm_start_graph] Không tìm thấy {p}")
        return False
    ckpt = torch.load(p, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    encoder_keys = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
    missing, unexpected = graph_encoder.load_state_dict(encoder_keys, strict=False)
    print(f"[warm_start_graph] Đã nạp thành công từ {p.name} (layers={len(encoder_keys)}, missing={len(missing)})")
    return True
