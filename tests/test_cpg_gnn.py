"""Test tầng CPG + model M4 — **không cần Joern, không cần GPU**.

Đây là lý do M4 vẫn tiến được dù máy không có Java: toàn bộ logic sau tầng
trích xuất (schema → ingest → store → GAT → pair head) được kiểm bằng đồ thị
tổng hợp đúng format mà `scripts/joern/export_cpg.py` sẽ ghi ra. Khi Joern có
thật, phần chưa kiểm chứng chỉ còn đúng một chỗ: script exporter đó ghi ra đúng
format này hay không.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.models.cpg_gnn.model import (
    CPGEncoder,
    CPGGNNCloneDetector,
    CPGGNNPairClassifier,
    TypedGATLayer,
)
from src.representations.cpg.graph import CPGraph, collate_pyg, graph_stats
from src.representations.cpg.ingest import build_from_export_dir, parse_cpg
from src.representations.cpg.schema import (
    N_EDGE_TYPES,
    N_NODE_TYPES,
    arg_count_feature,
    edge_type_id,
    node_type_id,
)
from src.representations.cpg.store import CPGStore


def make_synthetic_graph(rng: np.random.Generator, n_nodes: int = 12) -> dict:
    """CPG giả **đúng format JSON** mà `scripts/joern/export_cpg.py` sẽ ghi ra.

    Cố ý sinh ở tầng JSON chứ không dựng thẳng `CPGraph`: test đi đúng một đường
    thật là schema → ingest → store → model, nên lỗi ở tầng giữa không lọt qua.
    """
    labels = ["METHOD", "CALL", "IDENTIFIER", "LOCAL", "LITERAL"]
    nodes = []
    for i in range(n_nodes):
        label = labels[int(rng.integers(0, len(labels)))]
        nodes.append({"id": i, "label": label,
                      "n_args": int(rng.integers(0, 5)) if label == "CALL" else -1})
    nodes[0]["label"] = "METHOD"
    nodes[0]["n_args"] = -1
    edge_labels = ["AST", "CFG", "CALL", "REF", "REACHING_DEF"]
    edges = []
    for i in range(1, n_nodes):
        edges.append({"src": int(rng.integers(0, i)), "dst": i, "label": "AST"})
        if rng.random() < 0.4:
            dst = int(rng.integers(0, n_nodes))
            if dst != i:  # fixture sạch: tự cạnh để riêng cho test đếm
                edges.append({"src": i, "dst": dst,
                              "label": edge_labels[int(rng.integers(1, len(edge_labels)))]})
    return {"sample_id": "synthetic", "nodes": nodes, "edges": edges}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _graph_from(payload: dict) -> CPGraph:
    g, _ = parse_cpg(payload, max_nodes=0)
    return g


# ------------------------------------------------------------------- schema
def test_schema_ids_are_in_range():
    assert 0 <= node_type_id("METHOD") < N_NODE_TYPES
    assert 0 <= edge_type_id("CFG") < N_EDGE_TYPES


def test_unknown_node_label_falls_back_not_raises():
    """Joern đổi tên node type giữa các version. Ném lỗi ở giữa lúc parse 62k
    hàm là cách mất cả phiên; rơi về OTHER rồi đếm lại mới an toàn."""
    assert node_type_id("KHONG_CO_TRONG_SCHEMA") == node_type_id("OTHER")
    assert node_type_id("method") == node_type_id("METHOD")  # không phân biệt hoa thường


def test_unknown_edge_label_is_dropped():
    assert edge_type_id("INCLUDES") == -1


def test_arg_count_feature_is_bounded_and_monotonic():
    assert arg_count_feature(-1) == 0.0
    assert arg_count_feature(0) == 0.0
    assert 0 < arg_count_feature(2) < arg_count_feature(5)
    assert arg_count_feature(10_000) == 1.0  # chặn trên


# ------------------------------------------------------------------- ingest
def test_parse_cpg_orders_method_first_and_drops_selfloop(rng):
    payload = make_synthetic_graph(rng)
    payload["edges"].append({"src": 1, "dst": 1, "label": "AST"})
    payload["edges"].append({"src": 1, "dst": 2, "label": "INCLUDES"})
    g, st = parse_cpg(payload)

    assert st["has_method"] is True
    assert g.node_type[0] == node_type_id("METHOD")
    assert st["edges_dropped_selfloop"] == 1
    assert st["edges_dropped_unknown_label"] == 1
    src, dst = g.edge_index
    assert not (src == dst).any()


def test_parse_cpg_dedupes_edges(rng):
    payload = make_synthetic_graph(rng)
    payload["edges"].append(dict(payload["edges"][0]))  # trùng y hệt
    _, st = parse_cpg(payload)
    assert st["edges_kept"] < len(payload["edges"]) + 1


def test_parse_cpg_truncation_keeps_entry_point(rng):
    """Cắt theo thứ tự id sẽ cắt mất đuôi hàm (phần hay nhất). Phải cắt theo
    BFS từ METHOD để giữ phần luồng điều khiển."""
    payload = make_synthetic_graph(rng, n_nodes=40)
    g_full, _ = parse_cpg(payload, max_nodes=0)
    g_cut, st = parse_cpg(payload, max_nodes=10)

    assert g_cut.n_nodes == 10
    assert st["truncated"] is True
    assert g_cut.node_type[0] == node_type_id("METHOD")
    # đồ thị cắt vẫn phải có cạnh (không rơi thành 10 node rời rạc)
    assert g_cut.n_edges > 0
    assert g_full.n_nodes == 40


def test_parse_cpg_rejects_empty():
    with pytest.raises(ValueError):
        parse_cpg({"nodes": [], "edges": []})


def test_parse_cpg_reports_missing_file_as_missing_not_error(tmp_path, rng):
    payload = make_synthetic_graph(rng)
    (tmp_path / "s1.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "s2.json").write_text("{khong phai json", encoding="utf-8")
    graphs, ids, st = build_from_export_dir(tmp_path, ["s1", "s2", "s3"])

    assert ids == ["s1"]
    assert len(graphs) == 1
    assert st["n_missing"] == 1 and st["n_failed"] == 1


# -------------------------------------------------------------------- graph
def test_validate_catches_dangling_edge(rng):
    g = _graph_from(make_synthetic_graph(rng))
    broken = CPGraph(g.node_type, g.node_feat,
                     g.edge_index + 99, g.edge_type)
    with pytest.raises(ValueError):
        broken.validate()


def test_to_pyg_fields_are_tensors(rng):
    """`Data(x=numpy)` ⇒ `Batch.x` thành `list` ⇒ `nn.Linear` ném lỗi rất xa chỗ
    đặt nhầm. Khóa lại bằng test để không tái phát."""
    batch = collate_pyg([_graph_from(make_synthetic_graph(rng, 7))])
    for key in ("x", "node_type", "edge_index", "edge_type"):
        assert isinstance(getattr(batch, key), torch.Tensor), key
    assert batch.x.shape[1] == 1


def test_collate_pyg_concatenates_and_tracks_batch(rng):
    graphs = [_graph_from(make_synthetic_graph(rng, 8)) for _ in range(3)]
    batch = collate_pyg(graphs)
    assert int(batch.num_nodes) == sum(g.n_nodes for g in graphs)
    assert int(batch.num_graphs) == 3
    assert batch.batch.max().item() == 2


def test_permuted_graph_is_isomorphic(rng):
    g = _graph_from(make_synthetic_graph(rng, 15))
    perm = rng.permutation(g.n_nodes)
    p = g.permuted(perm)
    assert p.n_nodes == g.n_nodes
    assert p.n_edges == g.n_edges
    # tập cạnh (nhãn) phải bằng nhau dưới hoán vị
    assert sorted(zip(p.edge_type.tolist(), p.edge_index[0].tolist())) is not None
    assert set(p.edge_type.tolist()) == set(g.edge_type.tolist())


def test_graph_stats_exposes_missing_edge_types():
    """Cache đã build có thể là CPG *nửa vời* (chỉ AST+CFG). `graph_stats` phải
    nói thẳng loại cạnh nào vắng — đó là điều kiện để preflight chặn train."""
    import numpy as np
    from src.representations.cpg.graph import CPGraph, graph_stats
    from src.representations.cpg.schema import edge_type_id

    ast_only = CPGraph(
        node_type=np.array([0, 1, 2], dtype=np.int64),
        node_feat=np.zeros((3, 1), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_type=np.array([edge_type_id("AST"), edge_type_id("CFG")], dtype=np.int64),
    )
    s = graph_stats([ast_only])
    assert s["edge_type_counts"] == {"AST": 1, "CFG": 1}
    assert "REACHING_DEF" in s["edge_types_missing"]
    assert "CALL" in s["edge_types_missing"]
    assert "AST" not in s["edge_types_missing"]


def test_graph_stats_reports_size(rng):
    s = graph_stats([_graph_from(make_synthetic_graph(rng, 10)) for _ in range(4)])
    assert s["n_graphs"] == 4
    assert s["nodes_per_code_mean"] == 10
    assert s["nodes_total"] == 40


# -------------------------------------------------------------------- store
def test_store_roundtrip_is_exact(tmp_path, rng):
    """Cache `.npz` là nơi mất dữ liệu âm thầm (đọc lại phải **bằng đúng** graph
    gốc, không phải 'gần đúng')."""
    payloads = {f"s{i}": make_synthetic_graph(rng, 6 + i) for i in range(3)}
    export = tmp_path / "export"
    export.mkdir()
    for sid, p in payloads.items():
        (export / f"{sid}.json").write_text(json.dumps(p), encoding="utf-8")

    store = CPGStore.build(export, list(payloads), 0, cache_path=tmp_path / "c.npz",
                           verbose=False)
    assert len(store) == 3
    again = CPGStore.build(export, list(payloads), 0, cache_path=tmp_path / "c.npz",
                           verbose=False)
    for sid in payloads:
        a, b = store[sid], again[sid]
        assert np.array_equal(a.node_type, b.node_type)
        assert np.array_equal(a.node_feat, b.node_feat)
        assert np.array_equal(a.edge_index, b.edge_index)
        assert np.array_equal(a.edge_type, b.edge_type)


def test_store_collate_pairs(rng):
    graphs = [_graph_from(make_synthetic_graph(rng, 5)) for _ in range(2)]
    store = CPGStore(graphs, ["a", "b"])
    batch = store.collate(["a", "b"])
    assert int(batch.num_graphs) == 2
    assert store.has("a") and not store.has("zzz")


# -------------------------------------------------------------------- model
def test_typed_gat_layer_rejects_indivisible_dims():
    with pytest.raises(ValueError):
        TypedGATLayer(128, 10, 4, 128, 0.0)  # 10 % 4 != 0


def test_encoder_output_is_permutation_invariant(rng):
    """Cùng một đồ thị, đảo thứ tự node ⇒ vector đặc trưng phải **bằng nhau**.
    Lệch 1e-6 là bình thường (float + sum thứ tự khác)."""
    torch.manual_seed(0)
    model = CPGEncoder(hidden=32, num_layers=2, heads=4, dropout=0.0).eval()
    g = _graph_from(make_synthetic_graph(rng, 20))
    perm = rng.permutation(g.n_nodes)

    with torch.no_grad():
        h1 = model(collate_pyg([g]))
        h2 = model(collate_pyg([g.permuted(perm)]))
    assert torch.allclose(h1, h2, atol=1e-5)


def test_encoder_uses_edge_label(rng):
    """Đổi nhãn cạnh phải đổi vector đặc trưng — nếu không, `TypedGATLayer` chỉ
    là trang trí và mô hình đang chạy trên đồ thị vô nghĩa."""
    torch.manual_seed(0)
    model = CPGEncoder(hidden=32, num_layers=2, heads=4, dropout=0.0).eval()
    g = _graph_from(make_synthetic_graph(rng, 15))
    flipped = CPGraph(g.node_type, g.node_feat, g.edge_index,
                      (g.edge_type + 1) % N_EDGE_TYPES)
    with torch.no_grad():
        h1 = model(collate_pyg([g]))
        h2 = model(collate_pyg([flipped]))
    assert not torch.allclose(h1, h2, atol=1e-4)


def test_pair_classifier_forward_and_backward(rng):
    torch.manual_seed(0)
    model = CPGGNNPairClassifier(hidden=32, num_layers=2, heads=4)
    ga = [_graph_from(make_synthetic_graph(rng, 12)) for _ in range(2)]
    gb = [_graph_from(make_synthetic_graph(rng, 9)) for _ in range(2)]
    logits = model(collate_pyg(ga), collate_pyg(gb))
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    loss = torch.sigmoid(logits).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.param_count() > 0


def test_pair_head_is_symmetric_in_abs_diff(rng):
    """Đảo A/B chỉ đổi 2 khối đầu, khối `|hA−hB|` bất biến — head giống M1/M2/M3
    nên tín hiệu phải bất biến theo thứ tự cặp (DEC-016)."""
    torch.manual_seed(0)
    model = CPGGNNPairClassifier(hidden=32, num_layers=2, heads=4).eval()
    g = _graph_from(make_synthetic_graph(rng, 10))
    batch = collate_pyg([g])
    with torch.no_grad():
        h = model.encode_side(batch)
        pair = torch.cat([h, h, torch.abs(h - h)], dim=-1)
    assert torch.allclose(pair[:, 2 * model.enc_dim:], torch.zeros(1, model.enc_dim))


def test_clone_detector_requires_graph_lookup():
    model = CPGGNNPairClassifier(hidden=16, num_layers=1, heads=4)
    det = CPGGNNCloneDetector(model, graph_for=None)
    with pytest.raises(NotImplementedError):
        det.predict_proba([])


def test_clone_detector_predicts_probabilities(rng):
    torch.manual_seed(0)
    model = CPGGNNPairClassifier(hidden=16, num_layers=1, heads=4).eval()
    graphs = {str(i): _graph_from(make_synthetic_graph(rng, 8)) for i in range(3)}

    class _Pair:
        def __init__(self, a, b):
            self.code_a, self.code_b = a, b

    det = CPGGNNCloneDetector(model, graph_for=graphs.__getitem__, batch_size=2)
    probs = det.predict_proba([_Pair("0", "1"), _Pair("2", "0")])
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert det.predict([_Pair("0", "1")], threshold=0.0) == [1]
