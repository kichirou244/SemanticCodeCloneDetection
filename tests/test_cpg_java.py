"""Unit tests cho pipeline trích xuất và xử lý đồ thị CPG đối với Java."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_cpg_export import write_batch
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.ingest import parse_cpg
from src.representations.cpg.schema import node_type_id, edge_type_id


def test_write_batch_java(tmp_path: Path):
    batch_dir = tmp_path / "batch_java"
    samples = [
        {
            "sample_id": "java-sample-001",
            "code": "public int add(int a, int b) { return a + b; }",  # lone method -> needs wrapper
        },
        {
            "sample_id": "java-sample-002",
            "code": "public class Calculator { public int mul(int a, int b) { return a * b; } }",  # already class
        },
    ]

    index_file = write_batch(samples, batch_dir, language="java")
    assert index_file.exists()

    # Kiểm tra file sinh ra mang đuôi .java
    f0 = batch_dir / "00000.java"
    f1 = batch_dir / "00001.java"
    assert f0.exists()
    assert f1.exists()

    # Kiểm tra wrapper bọc quanh lone method
    f0_text = f0.read_text(encoding="utf-8")
    assert "public class Wrapper_00000" in f0_text
    assert "public int add(int a, int b)" in f0_text

    # Kiểm tra class có sẵn không bị bọc thừa
    f1_text = f1.read_text(encoding="utf-8")
    assert "public class Calculator" in f1_text
    assert "Wrapper_00001" not in f1_text

    # Kiểm tra index.tsv
    index_text = index_file.read_text(encoding="utf-8")
    assert "00000.java\tjava-sample-001" in index_text
    assert "00001.java\tjava-sample-002" in index_text


def test_java_cpg_json_parsing():
    # Mô phỏng CPG JSON do Joern export từ một hàm Java
    java_cpg_payload = {
        "sample_id": "bcb-java-sample",
        "nodes": [
            {"id": 1, "label": "METHOD", "code": "calculateSum", "n_args": -1},
            {"id": 2, "label": "PARAMETER", "code": "int a", "n_args": -1},
            {"id": 3, "label": "PARAMETER", "code": "int b", "n_args": -1},
            {"id": 4, "label": "BLOCK", "code": "{ return a + b; }", "n_args": -1},
            {"id": 5, "label": "CALL", "code": "a + b", "n_args": 2},
            {"id": 6, "label": "IDENTIFIER", "code": "a", "n_args": -1},
            {"id": 7, "label": "IDENTIFIER", "code": "b", "n_args": -1},
            {"id": 8, "label": "RETURN", "code": "return a + b;", "n_args": -1},
        ],
        "edges": [
            {"src": 1, "dst": 2, "label": "AST"},
            {"src": 1, "dst": 3, "label": "AST"},
            {"src": 1, "dst": 4, "label": "AST"},
            {"src": 4, "dst": 8, "label": "AST"},
            {"src": 8, "dst": 5, "label": "AST"},
            {"src": 5, "dst": 6, "label": "ARGUMENT"},
            {"src": 5, "dst": 7, "label": "ARGUMENT"},
            {"src": 1, "dst": 5, "label": "CFG"},
            {"src": 5, "dst": 8, "label": "CFG"},
        ],
    }

    graph, stats = parse_cpg(java_cpg_payload)
    assert graph.n_nodes == 8
    assert graph.n_edges == 9
    assert stats["nodes_kept"] == 8
    assert stats["has_method"] is True

    # Chuyển đổi sang PyG data
    pyg_data = graph.to_pyg()
    assert pyg_data.x.shape == (8, 1)
    assert pyg_data.edge_index.shape == (2, 9)
    assert pyg_data.edge_type.shape == (9,)


def test_java_cpg_model_forward():
    # Kiểm tra forward pass qua mô hình GNN với đồ thị Java
    g1 = CPGraph(
        node_type=np.array([node_type_id("METHOD"), node_type_id("CALL"), node_type_id("IDENTIFIER")], dtype=np.int64),
        node_feat=np.zeros((3, 1), dtype=np.float32),
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_type=np.array([edge_type_id("AST"), edge_type_id("CFG")], dtype=np.int64),
    )
    g2 = CPGraph(
        node_type=np.array([node_type_id("METHOD"), node_type_id("LITERAL")], dtype=np.int64),
        node_feat=np.zeros((2, 1), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_type=np.array([edge_type_id("AST")], dtype=np.int64),
    )

    batch_a = collate_pyg([g1, g2])
    batch_b = collate_pyg([g2, g1])

    model = CPGGNNPairClassifier(hidden=32, num_layers=2, heads=2, dropout=0.1)
    model.eval()

    with torch.no_grad():
        logits = model(batch_a, batch_b)
        assert logits.shape == (2,)
        assert torch.isfinite(logits).all()
