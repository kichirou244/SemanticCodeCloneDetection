"""Smoke test kiểm tra forward pass của cả 5 mô hình (M1..M5) trên dữ liệu cặp Java."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.loaders.base import ClonePairSample
from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.data.pairs.bcb_dataset import BCBPairDataset, collate_ast_pairs
from src.models.astnn.model import ASTNNPairClassifier
from src.models.cpg_gnn.model import CPGEncoder, CPGGNNPairClassifier
from src.models.hybrid.model import HybridPairClassifier
from src.models.transformer_pair import TransformerPairClassifier
from src.representations.ast.extract import N_STMT
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.schema import edge_type_id, node_type_id


class DummyTransformerEncoder(nn.Module):
    """Encoder giả lập RoBERTa để test nhanh không cần tải weights 500MB từ internet."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()
        self.embedding = nn.Embedding(1000, hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        emb = self.embedding(input_ids)
        return type("Output", (), {"last_hidden_state": emb})()


@pytest.fixture
def java_pairs() -> list[ClonePairSample]:
    # Sử dụng các cặp từ bcb_smoke.parquet nếu có, fallback danh sách mẫu
    smoke_path = Path("data/raw/bigclonebench/bcb_smoke.parquet")
    if smoke_path.exists():
        loader = BigCloneBenchLoader()
        return loader.load_parquet(smoke_path, max_samples=4)
    return [
        ClonePairSample(
            pair_id="p1",
            id1="1",
            id2="2",
            func1="public int add(int a, int b) { return a + b; }",
            func2="public int sum(int a, int b) { return a + b; }",
            label=1,
            language="java",
        ),
        ClonePairSample(
            pair_id="p2",
            id1="3",
            id2="4",
            func1="public void close() { System.exit(0); }",
            func2="public int multiply(int x, int y) { return x * y; }",
            label=0,
            language="java",
        ),
    ]


def test_m1_m2_transformer_smoke_forward(java_pairs):
    encoder = DummyTransformerEncoder(hidden_size=64)
    model = TransformerPairClassifier(hf_id="dummy", encoder=encoder)

    # Giả lập tokenized inputs
    batch_size = len(java_pairs)
    inputs_a = {"input_ids": torch.randint(0, 500, (batch_size, 16))}
    inputs_b = {"input_ids": torch.randint(0, 500, (batch_size, 16))}
    labels = torch.tensor([float(p.label) for p in java_pairs])

    logits = model(inputs_a, inputs_b)
    assert logits.shape == (batch_size,)
    assert torch.isfinite(logits).all()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_m3_astnn_smoke_forward(java_pairs):
    batch_a, batch_b, labels = collate_ast_pairs(java_pairs[:2], language="java")
    model = ASTNNPairClassifier(hidden=64, dropout=0.1, n_stmt=N_STMT)

    logits = model(batch_a, batch_b)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_m4_cpg_gnn_smoke_forward():
    g1 = CPGraph(
        node_type=np.array([node_type_id("METHOD"), node_type_id("CALL")], dtype=np.int64),
        node_feat=np.zeros((2, 1), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_type=np.array([edge_type_id("AST")], dtype=np.int64),
    )
    g2 = CPGraph(
        node_type=np.array([node_type_id("METHOD"), node_type_id("IDENTIFIER")], dtype=np.int64),
        node_feat=np.zeros((2, 1), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_type=np.array([edge_type_id("CFG")], dtype=np.int64),
    )

    batch_a = collate_pyg([g1, g2])
    batch_b = collate_pyg([g2, g1])
    labels = torch.tensor([1.0, 0.0])

    model = CPGGNNPairClassifier(hidden=32, num_layers=2, heads=2, dropout=0.1)
    logits = model(batch_a, batch_b)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_m5_hybrid_smoke_forward():
    text_enc = DummyTransformerEncoder(hidden_size=64)
    graph_enc = CPGEncoder(hidden=32, num_layers=2, heads=2, dropout=0.1)

    model = HybridPairClassifier(
        text_encoder=text_enc,
        graph_encoder=graph_enc,
        text_dim=64,
        head_hidden=64,
        head_dropout=0.1,
    )

    g1 = CPGraph(
        node_type=np.array([node_type_id("METHOD"), node_type_id("CALL")], dtype=np.int64),
        node_feat=np.zeros((2, 1), dtype=np.float32),
        edge_index=np.array([[0], [1]], dtype=np.int64),
        edge_type=np.array([edge_type_id("AST")], dtype=np.int64),
    )
    graph_batch_a = collate_pyg([g1, g1])
    graph_batch_b = collate_pyg([g1, g1])

    text_a = {"input_ids": torch.randint(0, 500, (2, 16))}
    text_b = {"input_ids": torch.randint(0, 500, (2, 16))}
    labels = torch.tensor([1.0, 0.0])

    logits = model(text_a, graph_batch_a, text_b, graph_batch_b)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)

    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
