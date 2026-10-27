"""Dataset & DataLoaders cho BigCloneBench Java pairs (dùng chung cho M1..M5)."""

from __future__ import annotations

from typing import Callable, Iterable
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.loaders.base import ClonePairSample
from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
from src.representations.cpg.graph import CPGraph, collate_pyg


class BCBPairDataset(Dataset):
    """Dataset chứa các cặp mã nguồn Java của BigCloneBench."""

    def __init__(self, pairs: list[ClonePairSample]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> ClonePairSample:
        return self.pairs[idx]


def collate_text_pairs(
    batch: list[ClonePairSample],
    tokenizer,
    max_length: int = 512,
    device: torch.device | None = None,
) -> tuple[dict, dict, torch.Tensor]:
    """Collate cho Transformer Siamese (M1 CodeBERT & M2 GraphCodeBERT)."""
    codes_a = [p.func1 for p in batch]
    codes_b = [p.func2 for p in batch]
    labels = [float(p.label) for p in batch]

    inputs_a = tokenizer(codes_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs_b = tokenizer(codes_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    label_tensor = torch.tensor(labels, dtype=torch.float32)

    if device is not None:
        inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
        inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
        label_tensor = label_tensor.to(device)

    return inputs_a, inputs_b, label_tensor


def collate_ast_pairs(
    batch: list[ClonePairSample],
    language: str = "java",
    device: torch.device | None = None,
    ast_store: Any | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """Collate cho ASTNN (M3), hỗ trợ tra cứu trực tiếp từ ASTTreeStore cache."""
    from src.models.astnn.model import to_device as astnn_to_device

    compiled_a = []
    compiled_b = []
    labels = []

    for p in batch:
        sid1 = str(p.id1)
        sid2 = str(p.id2)
        if ast_store is not None and sid1 in ast_store.index:
            c_a = ast_store[sid1]
        else:
            c_a = compile_tree(extract_tree(p.func1, language=language))

        if ast_store is not None and sid2 in ast_store.index:
            c_b = ast_store[sid2]
        else:
            c_b = compile_tree(extract_tree(p.func2, language=language))

        compiled_a.append(c_a)
        compiled_b.append(c_b)
        labels.append(float(p.label))

    batch_a_np = collate_trees(compiled_a)
    batch_b_np = collate_trees(compiled_b)

    dev = device or torch.device("cpu")
    batch_a = astnn_to_device(batch_a_np, dev)
    batch_b = astnn_to_device(batch_b_np, dev)
    label_tensor = torch.tensor(labels, dtype=torch.float32, device=dev)

    return batch_a, batch_b, label_tensor


def collate_cpg_pairs(
    batch: list[ClonePairSample],
    cpg_store: dict[str, CPGraph] | Path | str | None = None,
    max_nodes: int = 1500,
    device: torch.device | None = None,
) -> tuple[Any, Any, torch.Tensor]:
    """Collate cho CPG-GNN (M4), hỗ trợ tra cứu trực tiếp từ export JSON hoặc dictionary CPGraph."""
    from src.representations.cpg.ingest import parse_cpg_file

    graphs_a = []
    graphs_b = []
    labels = []

    cpg_dir = Path(cpg_store) if isinstance(cpg_store, (str, Path)) else None

    for p in batch:
        sid1 = str(p.id1)
        sid2 = str(p.id2)

        g_a = None
        g_b = None

        if isinstance(cpg_store, dict):
            g_a = cpg_store.get(sid1)
            g_b = cpg_store.get(sid2)
        elif cpg_dir is not None:
            f_a = cpg_dir / f"{sid1}.json"
            f_b = cpg_dir / f"{sid2}.json"
            if f_a.exists():
                g_a, _ = parse_cpg_file(f_a, max_nodes=max_nodes)
            if f_b.exists():
                g_b, _ = parse_cpg_file(f_b, max_nodes=max_nodes)

        # Fallback đồ thị an toàn nếu không tìm thấy CPG file
        if g_a is None:
            g_a = CPGraph(
                sample_id=sid1,
                node_type=np.zeros(1, dtype=np.int32),
                node_code=np.array(["missing"]),
                node_n_args=np.zeros(1, dtype=np.int32),
                edge_src=np.zeros(0, dtype=np.int32),
                edge_dst=np.zeros(0, dtype=np.int32),
                edge_type=np.zeros(0, dtype=np.int32),
            )
        if g_b is None:
            g_b = CPGraph(
                sample_id=sid2,
                node_type=np.zeros(1, dtype=np.int32),
                node_code=np.array(["missing"]),
                node_n_args=np.zeros(1, dtype=np.int32),
                edge_src=np.zeros(0, dtype=np.int32),
                edge_dst=np.zeros(0, dtype=np.int32),
                edge_type=np.zeros(0, dtype=np.int32),
            )

        graphs_a.append(g_a)
        graphs_b.append(g_b)
        labels.append(float(p.label))

    pyg_a = collate_pyg(graphs_a)
    pyg_b = collate_pyg(graphs_b)

    dev = device or torch.device("cpu")
    if dev.type == "cuda":
        pyg_a = pyg_a.to(dev)
        pyg_b = pyg_b.to(dev)

    label_tensor = torch.tensor(labels, dtype=torch.float32, device=dev)
    return pyg_a, pyg_b, label_tensor
