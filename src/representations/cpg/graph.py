"""Đồ thị CPG của **một hàm** (một sample trong dataset) — đơn vị nhỏ nhất.

Một code của POJ-104 là một hàm (hoặc prototype), nên một sample ↔ một CPG.
Đây là lựa chọn khác CPG-Python kiểu module-level: cặp clone được so trên **cấp
hàm**, và model M4 dùng graph-level embedding nên không cần biết cả file.

Chỉ giữ mảng numpy; chuyển sang đối tượng PyTorch Geometric chỉ lúc forward
(`to_pyg`) để tầng này không phụ thuộc thư viện graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.representations.cpg.schema import (
    EDGE_LABELS,
    FEATURE_DIM,
    N_EDGE_TYPES,
    N_NODE_TYPES,
)


@dataclass(frozen=True)
class CPGraph:
    """Một CPG: node type + node feature + cạnh có nhãn. Mảng đã được kiểm ở `ingest`."""

    node_type: np.ndarray   # int64 [N]   — index vào schema.NODE_LABELS
    node_feat: np.ndarray   # float32 [N, F]
    edge_index: np.ndarray  # int64 [2, E] — hàng 0 = nguồn, hàng 1 = đích
    edge_type: np.ndarray   # int64 [E]   — index vào schema.EDGE_LABELS

    @property
    def n_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_type.shape[0])

    def validate(self) -> None:
        """Bất biến mà mọi nguồn dữ liệu phải giữ. Ném lỗi sớm ở ingest
        thay vì để GATConv nhận shape sai và ra số NaN ở giữa lúc train."""
        n = self.n_nodes
        if self.node_type.dtype != np.int64 or self.node_feat.dtype != np.float32:
            raise ValueError("node_type phải int64, node_feat phải float32")
        if self.node_feat.shape != (n, FEATURE_DIM):
            raise ValueError(f"node_feat phải có shape ({n}, {FEATURE_DIM}), "
                             f"có {self.node_feat.shape}")
        if self.edge_type.dtype != np.int64 or self.edge_index.dtype != np.int64:
            raise ValueError("edge_index/edge_type phải int64")
        if self.edge_index.shape != (2, self.n_edges):
            raise ValueError(f"edge_index phải có shape (2, {self.n_edges}), "
                             f"có {self.edge_index.shape}")
        if n and (self.node_type.min() < 0 or self.node_type.max() >= N_NODE_TYPES):
            raise ValueError("node_type ngoài vocabulary schema")
        if self.n_edges and (self.edge_type.min() < 0
                             or self.edge_type.max() >= N_EDGE_TYPES):
            raise ValueError("edge_type ngoài vocabulary schema")
        if self.n_edges and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index trỏ tới node không tồn tại")

    def to_pyg(self):
        """Đối tượng `torch_geometric.data.Data` cho một graph.

        Mọi trường phải là **torch.Tensor**: nếu đưa `numpy.ndarray` vào `Data`,
        PyG coi nó là attribute thường và `Batch.from_data_list` gom thành
        `list` — lúc forward `nn.Linear` mới ném lỗi, khá xa chỗ đặt nhầm.
        """
        from torch_geometric.data import Data

        return Data(x=torch_f32(self.node_feat), node_type=torch_long(self.node_type),
                    edge_index=torch_long(self.edge_index),
                    edge_type=torch_long(self.edge_type))

    def permuted(self, perm: np.ndarray) -> "CPGraph":
        """Đổi thứ tự node (và cạnh theo). Dùng để test tính bất biến với hoán vị."""
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.shape[0], dtype=np.int64)
        src = inv[self.edge_index[0]]
        dst = inv[self.edge_index[1]]
        keep = (src != dst)  # tự cạnh biến thành cạnh tự vòng khi đổi thứ tự
        return CPGraph(self.node_type[perm], self.node_feat[perm],
                       np.stack([src[keep], dst[keep]]), self.edge_type[keep])

    def to_dict(self) -> dict[str, np.ndarray]:
        return {"node_type": self.node_type, "node_feat": self.node_feat,
                "edge_index": self.edge_index, "edge_type": self.edge_type}


def torch_long(arr: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))


def torch_f32(arr: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def collate_pyg(graphs: list[CPGraph]):
    """Ghép nhiều CPG thành một batch PyG (offset node index tự động)."""
    from torch_geometric.data import Batch

    for g in graphs:
        g.validate()
    return Batch.from_data_list([g.to_pyg() for g in graphs])


def graph_stats(graphs: list[CPGraph]) -> dict:
    """Thống kê kích thước đồ thị — dùng cho `preflight`/`diagnostic` của EXP-005.

    `nodes_per_code_mean` là con số quyết định batch_size: một hàm C/C++ trung
    bình có hàng trăm node, nên batch 32 cặp = hàng chục nghìn node mỗi step.
    """
    if not graphs:
        return {"n_graphs": 0}
    nn_ = np.array([g.n_nodes for g in graphs], dtype=np.int64)
    ne = np.array([g.n_edges for g in graphs], dtype=np.int64)
    # Đếm theo từng loại cạnh. `n_edges` lớn không bảo chứng có data-flow: một đồ
    # thị đầy AST + CFG vẫn có `edges_per_code_mean` rất cao và trông rất ổn,
    # trong khi đúng thứ phân biệt CPG với AST+CFG là REACHING_DEF.
    counts = np.zeros(N_EDGE_TYPES, dtype=np.int64)
    for g in graphs:
        if g.n_edges:
            counts += np.bincount(g.edge_type, minlength=N_EDGE_TYPES)
    return {
        "n_graphs": len(graphs),
        "nodes_per_code_mean": round(float(nn_.mean()), 1),
        "nodes_per_code_p95": round(float(np.percentile(nn_, 95)), 1),
        "nodes_per_code_max": int(nn_.max()),
        "edges_per_code_mean": round(float(ne.mean()), 1),
        "edges_per_code_max": int(ne.max()),
        "nodes_total": int(nn_.sum()),
        "edges_total": int(ne.sum()),
        "edge_type_counts": {EDGE_LABELS[i]: int(counts[i])
                              for i in range(N_EDGE_TYPES) if counts[i] > 0},
        "edge_types_missing": [EDGE_LABELS[i] for i in range(N_EDGE_TYPES)
                               if counts[i] == 0],
        "memory_note": "ước lượng 1 node ~1B feature + 1 int64 type; "
                       "xem CPGStore.stats['cache_mb'] sau khi build thật",
    }
