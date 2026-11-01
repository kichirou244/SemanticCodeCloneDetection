from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from src.representations.cpg.schema import FEATURE_DIM, N_EDGE_TYPES, N_NODE_TYPES


def to_device(batch, device: torch.device):
    return batch.to(device)


class TypedGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float) -> None:
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
    def __init__(self, hidden: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(N_NODE_TYPES, hidden)
        self.feat_lin = nn.Linear(FEATURE_DIM, hidden)
        self.in_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            TypedGATLayer(hidden, hidden, heads, hidden, dropout) for _ in range(num_layers)
        ])

    def forward(self, batch) -> torch.Tensor:
        x = self.in_norm(self.node_emb(batch.node_type) + self.feat_lin(batch.x))
        for layer in self.layers:
            x = x + layer(x, batch)
        return torch.cat([global_mean_pool(x, batch.batch), global_max_pool(x, batch.batch)], dim=-1)


class CPGGNNPairClassifier(nn.Module):
    def __init__(self, hidden: int = 128, num_layers: int = 3, heads: int = 4, dropout: float = 0.2) -> None:
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
        ha = self.encode_side(batch_a)
        hb = self.encode_side(batch_b)
        pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
