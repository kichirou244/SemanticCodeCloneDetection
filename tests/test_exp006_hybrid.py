"""Test model M5 Hybrid (GraphCodeBERT + CPG-GNN) — không cần GPU, không download HF.

Encoder text dùng stub (config.hidden_size + forward trả last_hidden_state) để test
không phụ thuộc mạng; graph dùng CPGEncoder thật trên đồ thị giả đúng format JSON
như test_cpg_gnn. Warm start được test bằng cách ghi checkpoint giả có khóa
`encoder.*` (đúng hình dạng state dict của M2/M4).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.cpg_gnn.model import CPGEncoder
from src.models.hybrid.model import (
    HybridPairClassifier,
    warm_start_graph,
    warm_start_text,
)
from src.representations.cpg.ingest import parse_cpg
from src.representations.cpg.graph import collate_pyg


class StubText(nn.Module):
    def __init__(self, dim: int = 12):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": dim})
        self.proj = nn.Linear(dim, dim)

    def forward(self, **kw):
        out = type("Out", (), {})
        out.last_hidden_state = self.proj(kw["input_ids"])
        return out


def make_graph(rng: np.random.Generator, n_nodes: int = 10):
    labels = ["METHOD", "CALL", "IDENTIFIER", "LOCAL", "LITERAL"]
    nodes = [{"id": i, "label": labels[int(rng.integers(0, len(labels)))],
              "n_args": int(rng.integers(0, 5)) if labels[int(rng.integers(0, len(labels)))] == "CALL" else -1}
             for i in range(n_nodes)]
    nodes[0] = {"id": 0, "label": "METHOD", "n_args": -1}
    edges = [{"src": int(rng.integers(0, i)), "dst": i, "label": "AST"} for i in range(1, n_nodes)]
    g, _ = parse_cpg({"nodes": nodes, "edges": edges}, max_nodes=0)
    return g


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def text_inputs(batch_size: int, seq: int, dim: int) -> dict:
    ids = torch.randn(batch_size, seq, dim)
    mask = torch.ones(batch_size, seq, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": mask}


def build(rng, freeze: bool = False, head_hidden: int = 24) -> HybridPairClassifier:
    return HybridPairClassifier(
        text_encoder=StubText(12), text_dim=12,
        graph_encoder=CPGEncoder(hidden=8, num_layers=2, heads=2, dropout=0.1),
        head_hidden=head_hidden, head_dropout=0.0, freeze_encoders=freeze)


def test_forward_shape_and_params(rng):
    model = build(rng)
    assert model.text_dim == 12
    assert model.graph_dim == 16
    assert model.enc_dim == 28
    assert model.param_count() > 0

    ga = collate_pyg([make_graph(rng), make_graph(rng), make_graph(rng)])
    gb = collate_pyg([make_graph(rng), make_graph(rng), make_graph(rng)])
    logits = model(text_inputs(3, 5, 12), ga, text_inputs(3, 5, 12), gb)
    assert logits.shape == (3,)
    assert logits.dtype == torch.float32
    assert float(logits.detach().std()) > 0


def test_backward_updates_only_trainable(rng):
    model = build(rng)
    y = torch.tensor([0.0, 1.0, 1.0])
    head_params = list(model.head.parameters())
    model.train()
    ga = collate_pyg([make_graph(rng), make_graph(rng), make_graph(rng)])
    gb = collate_pyg([make_graph(rng), make_graph(rng), make_graph(rng)])
    torch.nn.functional.binary_cross_entropy_with_logits(
        model(text_inputs(3, 5, 12), ga, text_inputs(3, 5, 12), gb), y).backward()
    assert all(p.grad is not None for p in head_params)
    assert any(p.grad is not None for p in model.graph_encoder.parameters())


def test_freeze_encoders_disables_grad_and_eval(rng):
    model = build(rng, freeze=True)
    assert model.trainable_param_count() == sum(p.numel() for p in model.head.parameters())
    model.train()
    assert not model.text_encoder.training
    assert not model.graph_encoder.training


def test_warm_start_text_and_graph_prefix_stripping(rng, tmp_path):
    """Checkpoint giả chứa khóa `encoder.*` (đúng hình dạng M2/M4) — warm_start
    phải nạp được vào encoder, bỏ qua `head.*`."""
    model = build(rng)
    sd = model.state_dict()
    text_ckpt = tmp_path / "text.pt"
    torch.save({"model_state": {f"encoder.{k.replace('text_encoder.', '')}": v
                                for k, v in sd.items() if k.startswith("text_encoder.")}},
               text_ckpt)
    graph_ckpt = tmp_path / "graph.pt"
    torch.save({"model_state": {f"encoder.{k.replace('graph_encoder.', '')}": v
                                for k, v in sd.items() if k.startswith("graph_encoder.")}},
               graph_ckpt)

    res_t = warm_start_text(model.text_encoder, str(text_ckpt), torch.device("cpu"))
    res_g = warm_start_graph(model.graph_encoder, str(graph_ckpt), torch.device("cpu"))
    assert res_t["loaded"] and res_t["missing"] == 0
    assert res_g["loaded"] and res_g["missing"] == 0


def test_warm_start_missing_file_is_skip_not_crash(rng, tmp_path):
    model = build(rng)
    res = warm_start_text(model.text_encoder, str(tmp_path / "none.pt"), torch.device("cpu"))
    assert res == {"loaded": False, "reason": "missing"}