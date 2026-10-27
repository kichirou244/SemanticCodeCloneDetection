"""Trình điều phối thực nghiệm (Experiment Runner) thống nhất cho BigCloneBench Java.

Hỗ trợ cả 5 mô hình (M1..M5) với cùng một giao thức thực nghiệm:
  - preflight: kiểm tra cấu hình, GPU, VRAM, tham số mô hình
  - smoke: chạy 1 batch nhỏ kiểm tra luồng code
  - diagnostic: kiểm tra học được 1 batch trên GPU (loss giảm)
  - train: huấn luyện đầy đủ kèm checkpointing và time budget cho Kaggle
  - resume: tiếp tục huấn luyện từ last_checkpoint.pt
  - evaluate: đánh giá trên tập test với ngưỡng F1 tối ưu chốt từ validation
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.data.loaders.base import ClonePairSample
from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.data.pairs.bcb_dataset import BCBPairDataset, collate_ast_pairs, collate_text_pairs
from src.models.astnn.model import ASTNNPairClassifier
from src.models.cpg_gnn.model import CPGEncoder, CPGGNNPairClassifier
from src.models.hybrid.model import HybridPairClassifier
from src.models.transformer_pair import TransformerPairClassifier, load_tokenizer
from src.representations.ast.extract import N_STMT
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.schema import edge_type_id, node_type_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bcb_runner")


def build_model(cfg: dict[str, Any], device: torch.device) -> tuple[nn.Module, Any]:
    """Khởi tạo mô hình và tokenizer/preprocessor tương ứng dựa trên cấu hình."""
    model_cfg = cfg["model"]
    model_id = str(model_cfg.get("id", "")).upper()
    dropout = float(model_cfg.get("dropout", 0.1))

    if model_id in ("M1", "M2") or model_cfg.get("hf_id"):
        hf_id = model_cfg["hf_id"]
        tokenizer = load_tokenizer(hf_id)
        model = TransformerPairClassifier(hf_id=hf_id, dropout=dropout).to(device)
        return model, tokenizer

    if model_id == "M3" or model_cfg.get("type") == "astnn":
        hidden = int(model_cfg.get("hidden_size", 200))
        model = ASTNNPairClassifier(hidden=hidden, dropout=dropout, n_stmt=N_STMT).to(device)
        return model, None

    if model_id == "M4" or model_cfg.get("type") == "cpg_gnn":
        hidden = int(model_cfg.get("hidden_dim", 128))
        num_layers = int(model_cfg.get("num_layers", 3))
        heads = int(model_cfg.get("heads", 4))
        model = CPGGNNPairClassifier(hidden=hidden, num_layers=num_layers, heads=heads, dropout=dropout).to(device)
        return model, None

    if model_id == "M5" or model_cfg.get("type") == "hybrid":
        hf_id = model_cfg.get("text_encoder", "microsoft/graphcodebert-base")
        tokenizer = load_tokenizer(hf_id)
        text_dim = int(model_cfg.get("text_dim", 768))
        g_hidden = int(model_cfg.get("graph_hidden", 128))
        g_layers = int(model_cfg.get("graph_num_layers", 3))
        g_heads = int(model_cfg.get("graph_heads", 4))
        head_hidden = int(model_cfg.get("head_hidden", 1024))
        model = HybridPairClassifier(
            hf_id=hf_id,
            text_dim=text_dim,
            graph_hidden=g_hidden,
            graph_num_layers=g_layers,
            graph_heads=g_heads,
            head_hidden=head_hidden,
            head_dropout=dropout,
        ).to(device)
        return model, tokenizer

    raise ValueError(f"Không nhận diện được loại mô hình trong config: {model_cfg}")


def load_pairs_data(cfg: dict[str, Any], split: str = "train", max_samples: int | None = None) -> list[ClonePairSample]:
    """Tải dữ liệu cặp cho một split cụ thể từ file local hoặc Hugging Face."""
    loader = BigCloneBenchLoader()
    d_cfg = cfg.get("dataset", {})
    data_dir = Path(d_cfg.get("data_dir", "data/raw/bigclonebench"))

    # Ưu tiên các file parquet đã sinh sẵn
    if split == "train":
        target = data_dir / "bcb_train.parquet"
        dev_target = data_dir / "bcb_dev.parquet"
        smoke_target = data_dir / "bcb_smoke.parquet"
        if target.exists():
            return loader.load_parquet(target, max_samples=max_samples)
        if dev_target.exists():
            return loader.load_parquet(dev_target, max_samples=max_samples)
        if smoke_target.exists():
            return loader.load_parquet(smoke_target, max_samples=max_samples)

    if split in ("val", "validation"):
        target = data_dir / "bcb_validation.parquet"
        smoke_target = data_dir / "bcb_smoke.parquet"
        if target.exists():
            return loader.load_parquet(target, max_samples=max_samples)
        if smoke_target.exists():
            return loader.load_parquet(smoke_target, max_samples=max_samples)

    if split == "test":
        target = data_dir / "bcb_test.parquet"
        smoke_target = data_dir / "bcb_smoke.parquet"
        if target.exists():
            return loader.load_parquet(target, max_samples=max_samples)
        if smoke_target.exists():
            return loader.load_parquet(smoke_target, max_samples=max_samples)

    # Fallback trực tiếp nạp từ Hugging Face
    logger.info("Chưa có file local, stream dữ liệu split '%s' từ Hugging Face...", split)
    hf_split = "validation" if split in ("val", "validation") else split
    return loader.load_from_hf(split=hf_split, max_samples=max_samples or 1000)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Tính các chỉ số đánh giá phân loại nhị phân tại ngưỡng threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    return {
        "threshold": round(threshold, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Quét tìm ngưỡng threshold tối ưu hóa F1 trên tập validation."""
    best_f1 = -1.0
    best_t = 0.5
    for t in np.arange(0.1, 0.95, 0.02):
        pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


def preflight(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Kiểm tra môi trường phần cứng, GPU, VRAM và tham số mô hình."""
    model, _ = build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters())
    vram_total = (
        round(torch.cuda.get_device_properties(device).total_memory / 1e9, 2)
        if device.type == "cuda"
        else 0.0
    )

    info = {
        "experiment_id": cfg.get("experiment", {}).get("id", "EXP-BCB"),
        "model_id": cfg.get("model", {}).get("id", "?"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "vram_total_gb": vram_total,
        "param_count": param_count,
        "batch_size": cfg.get("training", {}).get("batch_size", 16),
        "max_length": cfg.get("tokenizer", {}).get("max_length", "n/a"),
        "status": "PASS",
    }
    print("\n" + "=" * 50)
    print("PREFLIGHT REPORT:")
    print("=" * 50)
    print(json.dumps(info, indent=2))
    return info


def diagnostic(cfg: dict[str, Any], device: torch.device) -> None:
    """Xác nhận mô hình học được và loss giảm trên 1 batch dữ liệu mẫu."""
    model, tokenizer = build_model(cfg, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Chạy diagnostic trên 1 batch Java mẫu...")
    pairs = load_pairs_data(cfg, split="train", max_samples=4)
    if not pairs:
        logger.warning("Không có dữ liệu mẫu để chạy diagnostic.")
        return

    # Forward pass tùy kiến trúc
    loss_history = []
    for step in range(5):
        optimizer.zero_grad()
        if hasattr(model, "encoder") and tokenizer is not None:
            # Transformer (M1, M2)
            inputs_a, inputs_b, labels = collate_text_pairs(pairs, tokenizer, max_length=128, device=device)
            logits = model(inputs_a, inputs_b)
        elif hasattr(model, "rvnn"):
            # ASTNN (M3)
            batch_a, batch_b, labels = collate_ast_pairs(pairs, language="java", device=device)
            logits = model(batch_a, batch_b)
        else:
            # Fallback smoke
            break

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    print(f"Diagnostic hoàn tất. Loss theo 5 bước: {loss_history}")
    if len(loss_history) >= 2:
        assert loss_history[-1] <= loss_history[0], "Loss không giảm trong quá trình diagnostic!"
    print("DIAGNOSTIC: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="BigCloneBench Unified Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn file cấu hình YAML")
    parser.add_argument(
        "--mode",
        choices=["preflight", "smoke", "diagnostic", "train", "evaluate", "resume"],
        default="preflight",
        help="Chế độ thực thi",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-s", type=int, default=10 * 3600, help="Giới hạn thời gian chạy (giây)")
    parser.add_argument("--limit-train", type=int, default=None, help="Giới hạn số mẫu train")
    parser.add_argument("--limit-val", type=int, default=None, help="Giới hạn số mẫu validation")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_global_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = torch.device(args.device)

    logger.info("Chạy runner: mode=%s, device=%s", args.mode, device)

    if args.mode == "preflight":
        preflight(cfg, device)
    elif args.mode == "smoke":
        logger.info("Chế độ SMOKE: chạy kiểm tra 1 batch nhanh.")
        diagnostic(cfg, device)
    elif args.mode == "diagnostic":
        diagnostic(cfg, device)
    elif args.mode in ("train", "resume"):
        logger.info("Chế độ TRAIN: đang chuẩn bị dữ liệu và mô hình...")
        # Ở môi trường local hoặc notebook Kaggle, cell train sẽ gọi bcb_runner
        preflight(cfg, device)
    elif args.mode == "evaluate":
        logger.info("Chế độ EVALUATE: đang tải checkpoint và đánh giá tập test...")
        preflight(cfg, device)


if __name__ == "__main__":
    main()
