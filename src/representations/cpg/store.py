"""Kho CPG đã nạp + cache `.npz` cho M4.

Lý do có kho: Joern parse một hàm mất hàng giây (xem `reports/005`), nạp lại ở
mỗi epoch sẽ lặp lại đúng cái tốn kém nhất. `build()` nạp **một lần** từ thư mục
export của Joern rồi lưu `.npz`; các lần chạy sau (resume, evaluate) chỉ đọc
`.npz` — nhanh hơn parse hàng giờ.

Cấu trúc lưu **phẳng** + mảng con trỏ (giống `ASTTreeStore.save`): một file
`.npz` cho cả split, nạp lại bằng `index_add`-style offset. `max_nodes` được ghi
vào tên file vì đổi ngưỡng này là **đổi feature** — dùng lại cache cũ sẽ train
trên dữ liệu khác config mà không có cảnh báo nào.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.representations.cpg.graph import CPGraph, graph_stats
from src.representations.cpg.ingest import build_from_export_dir
from src.representations.cpg.schema import FEATURE_DIM, N_EDGE_TYPES, N_NODE_TYPES

_FLAT_KEYS = ("node_type", "node_feat", "edge_index", "edge_type")


class CPGStore:
    """Danh sách CPG của các hàm, tra theo `sample_id`."""

    def __init__(self, graphs: list[CPGraph], ids: list[str],
                 stats: dict | None = None) -> None:
        self.graphs = graphs
        self.ids = ids
        self.index = {sid: i for i, sid in enumerate(ids)}
        self.stats = stats or {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, sample_id: str) -> CPGraph:
        return self.graphs[self.index[sample_id]]

    def has(self, sample_id: str) -> bool:
        return sample_id in self.index

    def collate(self, sample_ids: list[str]):
        """Ghép các CPG của `sample_ids` thành một batch PyG."""
        from src.representations.cpg.graph import collate_pyg

        return collate_pyg([self.graphs[self.index[s]] for s in sample_ids])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, export_dir: Path, sample_ids: list[str], max_nodes: int = 0,
              cache_path: Path | None = None, verbose: bool = True) -> "CPGStore":
        if cache_path is not None and cache_path.exists():
            t0 = time.perf_counter()
            store = cls._load(cache_path)
            if verbose:
                print(f"[cpg] nap cache {cache_path.name}: {len(store)} do thi "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            return store

        t0 = time.perf_counter()
        graphs, ids, load_stats = build_from_export_dir(export_dir, sample_ids, max_nodes)
        load_s = time.perf_counter() - t0
        if verbose:
            print(f"[cpg] nap {len(graphs):,} do thi trong {load_s:.1f}s "
                  f"({load_stats['n_missing']} thieu file, "
                  f"{load_stats['n_failed']} parse loi)", flush=True)
        s = graph_stats(graphs)
        store = cls(graphs, ids, {**s, **load_stats, "load_seconds": round(load_s, 2),
                                  "max_nodes": max_nodes})
        if cache_path is not None:
            store.save(cache_path)
            store.stats["cache_mb"] = round(cache_path.stat().st_size / 2**20, 1)
            if verbose:
                print(f"[cpg] da luu cache {cache_path} "
                      f"({store.stats['cache_mb']} MiB)", flush=True)
        return store

    # ------------------------------------------------------------- save / load
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {}
        for key in _FLAT_KEYS:
            parts = [getattr(g, key) for g in self.graphs]
            if not parts:
                data[key] = np.zeros((2, 0) if key == "edge_index" else (0, 1),
                                     dtype=np.int64)
                continue
            # `edge_index` là (2, E) nên nối theo trục 1; còn lại nối theo trục 0
            # để node của mọi hàm nối thành một khối (offset tính ở `_load`).
            data[key] = np.concatenate(parts, axis=1 if key == "edge_index" else 0)
        data["node_feat"] = data["node_feat"].astype(np.float32)
        data["ids"] = np.asarray(self.ids)
        data["node_ptr"] = _ptr([g.n_nodes for g in self.graphs])
        data["edge_ptr"] = _ptr([g.n_edges for g in self.graphs])
        data["stats"] = np.asarray(json.dumps(self.stats))
        np.savez_compressed(path, **data)

    @classmethod
    def _load(cls, path: Path) -> "CPGStore":
        with np.load(path, allow_pickle=False) as z:
            ids = [str(s) for s in z["ids"]]
            nptr, eptr = z["node_ptr"], z["edge_ptr"]
            ntype, nfeat = z["node_type"], z["node_feat"]
            eindex, etype = z["edge_index"], z["edge_type"]
            stats = json.loads(str(z["stats"]))
            graphs: list[CPGraph] = []
            for i in range(len(ids)):
                n0, n1 = int(nptr[i]), int(nptr[i + 1])
                e0, e1 = int(eptr[i]), int(eptr[i + 1])
                graphs.append(CPGraph(
                    node_type=ntype[n0:n1],
                    node_feat=nfeat[n0:n1].reshape(-1, FEATURE_DIM),
                    edge_index=eindex[:, e0:e1],
                    edge_type=etype[e0:e1],
                ))
        # Stats phái sinh (kích thước đồ thị, coverage cạnh) thì **tính lại từ
        # graphs**, không tin blob: blob là ảnh chụp lúc ghi, nên một cache viết
        # bởi phiên bản cũ (thiếu `edge_types_missing`) sẽ khiến gate của
        # notebook ném KeyError thay vì báo đúng. Phần thống kê lúc nạp
        # (n_missing, n_failed, edges_dropped_*) thì không tính lại được nên
        # giữ nguyên từ blob; stats tính lại đè lên trên.
        return cls(graphs, ids, {**stats, **graph_stats(graphs)})


def _ptr(lens: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def schema_sizes() -> dict:
    return {"n_node_types": N_NODE_TYPES, "n_edge_types": N_EDGE_TYPES,
            "feature_dim": FEATURE_DIM}
