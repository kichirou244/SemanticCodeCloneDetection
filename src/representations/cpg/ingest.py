"""Nạp CPG từ JSON do `scripts/joern/export_cpg.py` sinh ra → `CPGraph`.

Định dạng JSON **do ta tự định nghĩa** (không phụ thuộc format nội bộ của Joern):
mỗi hàm một file, `{"sample_id", "nodes": [{id,label,n_args}], "edges": [{src,dst,label}]}`.
Nhờ vậy đổi backend trích xuất (Joern bản mới, hay bỏ Joern) chỉ đổi exporter, còn
`ingest` + model không đụng gì — và **quan trọng nhất**: phần này test được 100%
local, không cần cài Joern.

Mọi thao tác cắt bớt/bỏ (node vượt `max_nodes`, cạnh tự vòng, cạnh trùng, cạnh
nhãn lạ) đều **đếm** và trả về cùng stats — im lặng nuốt dữ liệu là lỗi tệ nhất
của tầng tiền xử lý.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from src.representations.cpg.graph import CPGraph
from src.representations.cpg.schema import arg_count_feature, edge_type_id, node_type_id

METHOD_LABEL = "METHOD"


def parse_cpg(payload: dict, max_nodes: int = 0) -> tuple[CPGraph, dict]:
    """JSON của một hàm → `(CPGraph, stats)`. `max_nodes > 0` là trần kích thước."""
    raw_nodes = payload.get("nodes") or []
    if not raw_nodes:
        raise ValueError("CPG rỗng: không có node nào")

    labels = [str(n.get("label", "")) for n in raw_nodes]
    order = _bfs_order(raw_nodes, payload.get("edges") or [], labels, max_nodes)
    n_raw = len(raw_nodes)
    truncated = len(order) < n_raw

    new_id = {raw_nodes[i]["id"]: new for new, i in enumerate(order)}
    node_type = np.array([node_type_id(labels[i]) for i in order], dtype=np.int64)
    node_feat = np.array(
        [[arg_count_feature(raw_nodes[i].get("n_args", -1))
          if labels[i].upper() == "CALL" else 0.0] for i in order],
        dtype=np.float32,
    )

    src_list, dst_list, type_list = [], [], []
    n_edge_unknown = n_edge_dangling = n_edge_selfloop = 0
    seen: set[tuple[int, int, int]] = set()
    for e in payload.get("edges") or []:
        t = edge_type_id(e.get("label", ""))
        if t < 0:
            n_edge_unknown += 1
            continue
        s, d = new_id.get(e.get("src")), new_id.get(e.get("dst"))
        if s is None or d is None:
            n_edge_dangling += 1
            continue
        if s == d:
            n_edge_selfloop += 1
            continue
        key = (s, d, t)
        if key in seen:
            continue
        seen.add(key)
        src_list.append(s)
        dst_list.append(d)
        type_list.append(t)

    graph = CPGraph(
        node_type=node_type,
        node_feat=node_feat,
        edge_index=np.array([src_list, dst_list], dtype=np.int64)
        if src_list else np.zeros((2, 0), dtype=np.int64),
        edge_type=np.array(type_list, dtype=np.int64) if type_list
        else np.zeros((0,), dtype=np.int64),
    )
    graph.validate()
    stats = {
        "nodes_raw": n_raw,
        "nodes_kept": graph.n_nodes,
        "truncated": truncated,
        "edges_kept": graph.n_edges,
        "edges_dropped_unknown_label": n_edge_unknown,
        "edges_dropped_dangling": n_edge_dangling,
        "edges_dropped_selfloop": n_edge_selfloop,
        "has_method": METHOD_LABEL in [labels[i].upper() for i in order],
    }
    return graph, stats


def parse_cpg_file(path: Path, max_nodes: int = 0) -> tuple[CPGraph, dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return parse_cpg(json.load(fh), max_nodes)


def _bfs_order(raw_nodes: list[dict], edges: list[dict], labels: list[str],
               max_nodes: int) -> list[int]:
    """Thứ tự node để cắt an toàn: BFS từ METHOD, giữ node khác theo thứ tự gốc.

    Cắt theo thứ tự gốc (id tăng dần) sẽ cắt mất đuôi hàm — phần hay nhất về
    ngữ nghĩa. BFS từ METHOD giữ lại phần từ entry point đi sâu, tức phần quyết
    định luồng điều khiển, và thứ tự này ổn định (deterministic) nên cache build
    lần sau cho kết quả y hệt.
    """
    n = len(raw_nodes)
    if max_nodes <= 0 or n <= max_nodes:
        return list(range(n))
    roots = [i for i, lab in enumerate(labels) if lab.upper() == METHOD_LABEL]
    if not roots:
        return list(range(max_nodes))
    children: dict[int, list[int]] = {}
    for i in range(n):
        children.setdefault(i, [])
    parent_of = {int(nd.get("id", -1)): i for i, nd in enumerate(raw_nodes)}
    for e in edges:
        p = parent_of.get(int(e.get("src", -1)))
        c = parent_of.get(int(e.get("dst", -1)))
        if p is not None and c is not None:
            children[p].append(c)
    seen, queue, out = {roots[0]}, deque([roots[0]]), []
    while queue and len(out) < max_nodes:
        cur = queue.popleft()
        out.append(cur)
        for nxt in children.get(cur, ()):
            if nxt not in seen and len(out) < max_nodes:
                seen.add(nxt)
                queue.append(nxt)
    for i in range(n):  # phần còn lại, giữ thứ tự gốc
        if len(out) >= max_nodes:
            break
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_from_export_dir(export_dir: Path, sample_ids: list[str],
                          max_nodes: int = 0) -> tuple[list[CPGraph], list[str], dict]:
    """Nạp CPG cho nhiều hàm từ thư mục export (`<sample_id>.json`).

    Thiếu file = mẫu đó **không** dùng được, không phải lỗi: Joern parse lỗi với
    một hàm C++ cũ là chuyện thường. Danh sách thiếu được trả về để caller quyết
    định (bỏ cặp có thiếu, hay dừng cả lần chạy).
    """
    graphs, loaded, missing, failed = [], [], [], []
    stats_acc = {"edges_dropped_unknown_label": 0, "edges_dropped_dangling": 0,
                 "edges_dropped_selfloop": 0, "truncated": 0, "nodes_raw": 0}
    for sid in sample_ids:
        path = Path(export_dir) / f"{sid}.json"
        if not path.exists():
            missing.append(sid)
            continue
        try:
            g, st = parse_cpg_file(path, max_nodes)
        except (ValueError, json.JSONDecodeError, OSError):
            failed.append(sid)
            continue
        graphs.append(g)
        loaded.append(sid)
        for k in stats_acc:
            stats_acc[k] += int(st.get(k, 0))
    return graphs, loaded, {"n_loaded": len(graphs), "n_missing": len(missing),
                            "n_failed": len(failed), "missing": missing[:20],
                            "failed": failed[:20], **stats_acc}
