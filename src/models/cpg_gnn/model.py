"""M4 — CPG-GNN trên CPG của từng hàm (POJ-104, C++).

Kiến trúc (bám theo hướng CPG-GNN của Li et al. 2019, rút gọn cho ngân sách của
dự án — xem `docs/decision_log.md` DEC-024):

  1. **Node embedding** = embedding(nhãn node) ⊕ Linear(feature scalar).
  2. **GAT có điều kiện nhãn cạnh**: mỗi cạnh được mã hóa thành vector và cộng
     vào hệ số attention. Nhờ vậy `CALL` khác `CFG` khác `REACHING_DEF` — bỏ phần
     này thì đồ thị chỉ còn là GAT trên cây, mất đúng thứ làm CPG có ý nghĩa.
  3. **Readout** = `[mean ; max]` → vector đặc trưng hàm.
  4. **Pair head** giống hệt M1/M2/M3 (DEC-016): `[hA ; hB ; |hA−hB|] → MLP → logit`.

Cố ý **không** dùng thư viện GNN nào ngoài `torch_geometric.nn.GATConv`: phần
đáng giá nhất (GAT) có sẵn, còn phần còn lại chỉ là `index_add` mà viết tay rõ
hơn là debug.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from src.representations.cpg.schema import FEATURE_DIM, N_EDGE_TYPES, N_NODE_TYPES


def to_device(batch, device: torch.device):
    """Đưa một batch PyG sang `device` (Batch đã gộp sẵn node/edge theo batch)."""
    return batch.to(device)


class TypedGATLayer(nn.Module):
    """Một lớp GAT đa đầu, mỗi cạnh mang nhãn riêng."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int,
                 dropout: float) -> None:
        super().__init__()
        if out_dim % heads:
            raise ValueError(f"out_dim={out_dim} phai chia het cho heads={heads}")
        self.edge_emb = nn.Embedding(N_EDGE_TYPES, edge_dim)
        self.conv = GATConv(in_dim, out_dim // heads, heads=heads, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch) -> torch.Tensor:
        e = self.edge_emb(batch.edge_type)
        return self.dropout(self.norm(self.conv(x, batch.edge_index, e)))


class CPGEncoder(nn.Module):
    """Mã hóa một CPG (một hàm) thành vector `out_dim = 2 * hidden`."""

    def __init__(self, hidden: int = 128, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(N_NODE_TYPES, hidden)
        self.feat_lin = nn.Linear(FEATURE_DIM, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Tầng đầu nhận `hidden`, các tầng sau nhận `hidden`; giữ nguyên kích
            # thước để dùng chung một cách đọc ở forward.
            self.layers.append(TypedGATLayer(hidden, hidden, heads, hidden, dropout))

    def forward(self, batch) -> torch.Tensor:
        x = self.node_emb(batch.node_type) + self.feat_lin(batch.x)
        x = self.in_norm(x)
        for layer in self.layers:
            x = x + layer(x, batch)  # residual: 3 tầng mà hidden=128 thì dễ chết gradient
        return torch.cat([global_mean_pool(x, batch.batch),
                          global_max_pool(x, batch.batch)], dim=-1)


class CPGGNNPairClassifier(nn.Module):
    """M4 CPG-GNN cho bài toán clone *pair* — head giống M1/M2/M3 (DEC-016)."""

    def __init__(self, hidden: int = 128, num_layers: int = 3, heads: int = 4,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = CPGEncoder(hidden, num_layers, heads, dropout)
        self.enc_dim = 2 * hidden
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * self.enc_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_side(self, batch) -> torch.Tensor:
        return self.encoder(batch)

    def forward(self, batch_a, batch_b) -> torch.Tensor:
        h_a = self.encode_side(batch_a)
        h_b = self.encode_side(batch_b)
        pair = torch.cat([h_a, h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CPGGNNCloneDetector:
    """Adapter theo `src.core.interfaces.CloneDetector` — dựng sau khi có CPG cache.

    Interface nhận **chuỗi code**, còn CPG store tra theo `sample_id`; nên adapter
    này nhận sẵn hàm tra CPG (`graph_for(code)`) do caller (CPGStore + tầng
    trích xuất) cấp. Không có hàm tra ⇒ ném lỗi nói rõ nguyên nhân thay vì trả
    số 0.5 giả.
    """

    BACKBONE = "gat"

    def __init__(self, model: CPGGNNPairClassifier, graph_for, device=None,
                 batch_size: int = 64) -> None:
        self.model = model
        self._graph_for = graph_for
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size

    def predict_proba(self, pairs: list) -> list[float]:
        from src.representations.cpg.graph import collate_pyg

        if self._graph_for is None:
            raise NotImplementedError(
                "CPGGNNCloneDetector cần hàm tra CPG theo mã nguồn; CPG phải được "
                "trích xuất bằng Joern trước (xem notebooks/exp005_cpg_extract_kaggle.ipynb)")
        self.model.eval()
        probs: list[float] = []
        with torch.no_grad():
            for chunk in _chunks(pairs, self.batch_size):
                ga = [self._graph_for(p.code_a) for p in chunk]
                gb = [self._graph_for(p.code_b) for p in chunk]
                logits = self.model(collate_pyg(ga).to(self.device),
                                    collate_pyg(gb).to(self.device))
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return probs

    def predict(self, pairs: list, threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(pairs)]


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
