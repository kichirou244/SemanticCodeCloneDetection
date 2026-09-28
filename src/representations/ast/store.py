"""Kho cây AST đã compile + collation theo batch cho M3 ASTNN.

Vì sao cần kho riêng: `compile_tree` chạy vòng lặp Python O(n·depth) cho mỗi code.
Gọi lại mỗi epoch sẽ tốn ~3× số code không cần thiết. `build()` làm **một lần**
rồi lưu `.npz`; các lần chạy sau (resume, evaluate) chỉ nạp lại.

Bộ nhớ: ~13 KB/code (đo trên POJ-104) → 41.491 codes ≈ 0.55 GB cho train split.
Kaggle có 32 GB RAM nên chấp nhận được; xem `MEMORY_NOTE`.

Về cache phẳng: `collate_trees` chỉ ghép nối mảng của từng code rồi cộng offset
(đã kiểm ở `extract.py:344-351`). Nên lưu `.npz` dạng một mảng lớn + mảng con trỏ
tương đương **đúng bằng** kết quả `collate_trees` cho mọi split, và nạp lại nhanh
hơn parse khoảng 20×. `s_level_ptr` không lưu vì `collate_trees` tự tính lại từ
`s_depth`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.representations.ast.extract import collate_trees, compile_tree, extract_tree, get_parser

MEMORY_NOTE = "~13 KB/code (do tren POJ-104) -> 41k codes ~0.55 GB cho train split"

# `s_order` cũng lưu phẳng: thứ tự global của nó phụ thuộc batch, nên khi nạp lại
# ta khôi phục `s_order` theo đúng thứ tự DFS cục bộ (depth giảm dần, stable)
# rồi để `collate_trees` sắp xếp lại theo batch.
_FLAT_KEYS = ("s_label", "s_parent", "s_depth", "anc_s", "anc_c", "top")


class ASTTreeStore:
    """Danh sách cây đã compile, tra theo `sample_id`."""

    def __init__(
        self,
        trees: list[dict[str, np.ndarray]],
        ids: list[str],
        stats: dict | None = None,
    ) -> None:
        self.trees = trees
        self.ids = ids
        self.index = {sid: i for i, sid in enumerate(ids)}
        self.stats = stats or {}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, sample_id: str) -> dict[str, np.ndarray]:
        return self.trees[self.index[sample_id]]

    def collate(self, sample_ids: list[str]) -> dict[str, np.ndarray]:
        return collate_trees([self.trees[self.index[s]] for s in sample_ids])

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        codes: dict[str, str],
        language: str = "cpp",
        cache_path: Path | None = None,
        verbose: bool = True,
    ) -> "ASTTreeStore":
        if cache_path is not None and cache_path.exists():
            t0 = time.perf_counter()
            store = cls._load(cache_path)
            if verbose:
                print(f"[ast] nap cache {cache_path.name}: {len(store)} codes "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            return store

        ids = list(codes)
        parser = get_parser(language)
        t0 = time.perf_counter()
        trees: list[dict[str, np.ndarray]] = []
        for i, sid in enumerate(ids):
            trees.append(compile_tree(extract_tree(codes[sid], parser, language)))
            if verbose and (i + 1) % 10000 == 0:
                el = time.perf_counter() - t0
                print(f"[ast] {i + 1:,}/{len(ids):,} codes  {el:.1f}s "
                      f"({(i + 1) / el:,.0f}/s)", flush=True)
        build_s = time.perf_counter() - t0

        store = cls(trees, ids, _aggregate_stats(trees, build_s))
        if verbose:
            s = store.stats
            print(f"[ast] compile xong {len(ids):,} codes trong {build_s:.1f}s "
                  f"({len(ids) / max(build_s, 1e-9):,.0f} codes/s)", flush=True)
            print(f"[ast] stmt mean={s['n_stmt_mean']} max={s['n_stmt_max']} | "
                  f"top mean={s['n_top_mean']} max={s['n_top_max']} | "
                  f"depth max={s['depth_max']} | {s['memory_note']}", flush=True)
        if cache_path is not None:
            store.save(cache_path)
            if verbose:
                print(f"[ast] da luu cache {cache_path} "
                      f"({cache_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
        return store

    # ------------------------------------------------------------- save / load
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {
            k: np.concatenate([t[k] for t in self.trees]) for k in _FLAT_KEYS
        }
        data["ids"] = np.asarray(self.ids)
        data["stmt_ptr"] = _ptr([int(t["n_stmt"]) for t in self.trees])
        data["anc_ptr"] = _ptr([len(t["anc_s"]) for t in self.trees])
        data["top_ptr"] = _ptr([int(t["n_top"]) for t in self.trees])
        data["stats"] = np.asarray(json.dumps(self.stats))
        np.savez_compressed(path, **data)

    @classmethod
    def _load(cls, path: Path) -> "ASTTreeStore":
        with np.load(path, allow_pickle=False) as z:
            ids = [str(s) for s in z["ids"]]
            ptr, aptr, tptr = z["stmt_ptr"], z["anc_ptr"], z["top_ptr"]
            s_label, s_parent = z["s_label"], z["s_parent"]
            s_depth, anc_s, anc_c, top = z["s_depth"], z["anc_s"], z["anc_c"], z["top"]
            stats = json.loads(str(z["stats"]))

            trees: list[dict[str, np.ndarray]] = []
            for i in range(len(ids)):
                s0, s1 = int(ptr[i]), int(ptr[i + 1])
                a0, a1 = int(aptr[i]), int(aptr[i + 1])
                t0, t1 = int(tptr[i]), int(tptr[i + 1])
                depth = s_depth[s0:s1]
                trees.append({
                    "s_label": s_label[s0:s1],
                    "s_parent": s_parent[s0:s1],
                    "s_depth": depth,
                    "s_order": np.argsort(-depth, kind="stable").astype(np.int64),
                    "anc_s": anc_s[a0:a1],
                    "anc_c": anc_c[a0:a1],
                    "top": top[t0:t1],
                    "n_stmt": np.int32(s1 - s0),
                    "n_top": np.int32(t1 - t0),
                })
        return cls(trees, ids, stats)


def _ptr(lens: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def _aggregate_stats(trees: list[dict[str, np.ndarray]], build_s: float) -> dict:
    n_stmt = [int(t["n_stmt"]) for t in trees]
    n_top = [int(t["n_top"]) for t in trees]
    anc = [len(t["anc_s"]) for t in trees]
    depth = [int(t["s_depth"].max()) for t in trees]
    n = len(trees) or 1
    return {
        "n_codes": len(trees),
        "build_seconds": round(build_s, 2),
        "codes_per_s": round(len(trees) / max(build_s, 1e-9), 1),
        "n_stmt_mean": round(sum(n_stmt) / n, 1),
        "n_stmt_max": max(n_stmt, default=0),
        "n_stmt_p95": round(float(np.percentile(n_stmt, 95)), 1),
        "n_top_mean": round(sum(n_top) / n, 1),
        "n_top_max": max(n_top, default=0),
        "anc_pairs_mean": round(sum(anc) / n, 1),
        "depth_max": max(depth, default=0),
        "memory_note": MEMORY_NOTE,
    }
