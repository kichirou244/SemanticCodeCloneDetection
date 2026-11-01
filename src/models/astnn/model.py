from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.representations.ast.extract import N_LEAF, N_STMT_CPP


def to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("s_label", "s_parent", "s_order", "anc_s", "anc_c", "top_idx")
    out = {k: torch.as_tensor(batch[k], dtype=torch.long, device=device) for k in keys}
    out["s_level_ptr"] = torch.as_tensor(batch["s_level_ptr"], dtype=torch.long, device=device)
    out["top_mask"] = torch.as_tensor(batch["top_mask"], dtype=torch.float32, device=device)
    out["n_levels"] = int(len(batch["s_level_ptr"]) - 1)
    out["batch_size"] = int(batch["batch_size"])
    return out


class VectorizedRvNN(nn.Module):
    def __init__(self, hidden: int = 200, n_stmt: int = N_STMT_CPP):
        super().__init__()
        self.hidden = hidden
        self.stmt_emb = nn.Embedding(n_stmt, hidden)
        self.leaf_emb = nn.Embedding(N_LEAF, hidden)
        self.W_s = nn.Linear(hidden, hidden)
        self.W_x = nn.Linear(hidden, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        s_label = b["s_label"]
        s_parent = b["s_parent"]
        order = b["s_order"]
        anc_s = b["anc_s"]
        anc_c = b["anc_c"]
        level_ptr = b["s_level_ptr"]
        n_stmt = s_label.shape[0]

        if anc_s.numel() > 0:
            leaf_h = self.leaf_emb(anc_c)
            x = torch.zeros(n_stmt, self.hidden, dtype=leaf_h.dtype, device=leaf_h.device)
            x = x.index_add(0, anc_s, self.W_x(leaf_h))
        else:
            x = torch.zeros(n_stmt, self.hidden, dtype=self.W_s.weight.dtype, device=s_label.device)

        x = x + self.W_s(self.stmt_emb(s_label))

        h_acc = torch.zeros(n_stmt + 1, self.hidden, dtype=x.dtype, device=x.device)
        h_out = torch.zeros(n_stmt, self.hidden, dtype=x.dtype, device=x.device)
        for j in range(b["n_levels"]):
            lo = int(level_ptr[j])
            hi = int(level_ptr[j + 1])
            if hi <= lo:
                continue
            nodes = order[lo:hi]
            h = self.act(x.index_select(0, nodes) + self.W_h(h_acc.index_select(0, nodes)))
            h_out = h_out.index_copy(0, nodes, h)
            par = s_parent.index_select(0, nodes)
            par = torch.where(par == nodes, torch.full_like(par, n_stmt), par)
            h_acc = h_acc.index_add(0, par, h)
        return h_out


class ASTNNPairClassifier(nn.Module):
    def __init__(self, hidden: int = 200, dropout: float = 0.1, n_stmt: int = N_STMT_CPP):
        super().__init__()
        self.hidden = hidden
        self.rvnn = VectorizedRvNN(hidden, n_stmt=n_stmt)
        self.enc_dim = 2 * (hidden // 2)
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * self.enc_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_side(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        h_stmt = self.rvnn(b)
        seq = h_stmt.index_select(0, b["top_idx"].reshape(-1)).view(
            b["batch_size"], b["top_idx"].shape[1], self.hidden
        )
        lengths = b["top_mask"].sum(dim=1).long().clamp(min=1).cpu()
        packed = pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return torch.cat([h_n[0], h_n[1]], dim=-1)

    def forward(self, batch_a: dict[str, torch.Tensor], batch_b: dict[str, torch.Tensor]) -> torch.Tensor:
        ha = self.encode_side(batch_a)
        hb = self.encode_side(batch_b)
        pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
