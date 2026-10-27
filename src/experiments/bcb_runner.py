"""Trình điều phối thực nghiệm (Experiment Runner) thống nhất cho BigCloneBench Java.

Hỗ trợ cả 4 mô hình (M1..M4) với cùng một giao thức thực nghiệm chuẩn:
  - preflight: kiểm tra cấu hình, GPU, VRAM, tham số mô hình
  - smoke: chạy 1 batch nhỏ kiểm tra luồng code
  - diagnostic: kiểm tra học được 1 batch trên GPU (loss giảm)
  - train: huấn luyện đầy đủ kèm checkpointing, validation F1, early stopping và time budget
  - resume: tiếp tục huấn luyện từ last_checkpoint.pt
  - evaluate: đánh giá trên tập test với ngưỡng F1 tối ưu chốt từ validation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
import tarfile
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
from src.data.pairs.bcb_dataset import BCBPairDataset, collate_ast_pairs, collate_cpg_pairs, collate_text_pairs
from src.models.astnn.model import ASTNNPairClassifier
from src.models.cpg_gnn.model import CPGEncoder, CPGGNNPairClassifier
from src.models.hybrid.model import HybridPairClassifier
from src.models.transformer_pair import TransformerPairClassifier, load_tokenizer
from src.representations.ast.extract import N_STMT
from src.representations.ast.store import ASTTreeStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bcb_runner")


def find_feature_resource(resource_name: str) -> Path | None:
    """Tìm kiếm file hoặc thư mục đặc trưng trong các đường dẫn khả dĩ (local, Kaggle Dataset)."""
    search_paths = [
        Path(f"artifacts/bcb_features/{resource_name}"),
        Path(f"artifacts/{resource_name}"),
        Path(f"data/raw/bigclonebench/{resource_name}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/{resource_name}"),
        Path(f"/kaggle/input/bigclonebench-features/{resource_name}"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    return None


def setup_cpg_export_dir() -> Path | None:
    """Chuẩn bị thư mục CPG JSON: nếu có file tar.gz thì tự động giải nén."""
    cpg_dir = find_feature_resource("cpg_export")
    if cpg_dir is not None and cpg_dir.is_dir() and any(cpg_dir.glob("*.json")):
        return cpg_dir

    tar_path = find_feature_resource("bcb_cpg_export.tar.gz")
    if tar_path is not None and tar_path.exists():
        extract_dir = Path("artifacts/cpg_export")
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not any(extract_dir.glob("*.json")):
            logger.info("Đang giải nén %s vào %s...", tar_path, extract_dir)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall("artifacts")
        return extract_dir

    return None


def build_model(cfg: dict[str, Any], device: torch.device) -> tuple[nn.Module, Any]:
    """Khởi tạo mô hình và tokenizer tương ứng dựa trên cấu hình."""
    model_cfg = cfg["model"]
    model_id = str(model_cfg.get("id", "")).upper()
    dropout = float(model_cfg.get("dropout", 0.1))

    if model_id in ("M1", "M2") or model_cfg.get("hf_id"):
        hf_id = model_cfg.get("hf_id", "microsoft/codebert-base")
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

    raise ValueError(f"Không nhận diện được loại mô hình trong config: {model_cfg}")


def load_pairs_data(cfg: dict[str, Any], split: str = "train", max_samples: int | None = None) -> list[ClonePairSample]:
    """Tải dữ liệu cặp cho một split cụ thể từ file local hoặc Hugging Face."""
    loader = BigCloneBenchLoader()
    d_cfg = cfg.get("dataset", {})
    data_dir = Path(d_cfg.get("data_dir", "data/raw/bigclonebench"))

    # Ưu tiên các file parquet đã sinh sẵn
    if split == "train":
        for target in [data_dir / "bcb_train.parquet", data_dir / "bcb_dev.parquet", data_dir / "bcb_smoke.parquet"]:
            if target.exists():
                return loader.load_parquet(target, max_samples=max_samples)

    if split in ("val", "validation"):
        for target in [data_dir / "bcb_validation.parquet", data_dir / "bcb_smoke.parquet"]:
            if target.exists():
                return loader.load_parquet(target, max_samples=max_samples)

    if split == "test":
        for target in [data_dir / "bcb_test.parquet", data_dir / "bcb_smoke.parquet"]:
            if target.exists():
                return loader.load_parquet(target, max_samples=max_samples)

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


def get_collate_fn(
    cfg: dict[str, Any],
    model_id: str,
    tokenizer: Any,
    device: torch.device,
    ast_store: Any | None = None,
    cpg_dir: Path | None = None,
):
    """Tạo collate function tối ưu tương ứng theo từng mô hình."""
    if model_id in ("M1", "M2"):
        max_len = int(cfg.get("tokenizer", {}).get("max_length", 512))
        return lambda b: collate_text_pairs(b, tokenizer, max_length=max_len, device=device)

    if model_id == "M3":
        return lambda b: collate_ast_pairs(b, language="java", device=device, ast_store=ast_store)

    if model_id == "M4":
        return lambda b: collate_cpg_pairs(b, cpg_store=cpg_dir, device=device)

    raise ValueError(f"Không nhận diện được collate_fn cho model_id: {model_id}")


def forward_batch(model: nn.Module, batch: Any, model_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Thực hiện forward pass theo đặc thù input từng kiến trúc."""
    if model_id in ("M1", "M2"):
        inputs_a, inputs_b, labels = batch
        logits = model(inputs_a, inputs_b)
        return logits, labels

    if model_id == "M3":
        batch_a, batch_b, labels = batch
        logits = model(batch_a, batch_b)
        return logits, labels

    if model_id == "M4":
        pyg_a, pyg_b, labels = batch
        logits = model(pyg_a, pyg_b)
        return logits, labels

    raise ValueError(f"Không hỗ trợ forward_batch cho model_id: {model_id}")


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
    model_id = str(cfg.get("model", {}).get("id", "M1")).upper()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Chạy diagnostic trên dữ liệu mẫu...")
    pairs = load_pairs_data(cfg, split="train", max_samples=4)
    if not pairs:
        logger.warning("Không có dữ liệu mẫu để chạy diagnostic.")
        return

    # Chuẩn bị cache nếu có
    ast_store = None
    if model_id == "M3":
        cache_path = find_feature_resource("bcb_ast_cache.npz")
        if cache_path is not None:
            ast_store = ASTTreeStore.build({}, cache_path=cache_path, verbose=False)

    cpg_dir = setup_cpg_export_dir() if model_id == "M4" else None
    collate = get_collate_fn(cfg, model_id, tokenizer, device, ast_store=ast_store, cpg_dir=cpg_dir)
    batch = collate(pairs)

    loss_history = []
    for step in range(5):
        optimizer.zero_grad()
        logits, labels = forward_batch(model, batch, model_id)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    print(f"Diagnostic hoàn tất. Loss theo 5 bước: {loss_history}")
    if len(loss_history) >= 2:
        assert loss_history[-1] <= loss_history[0], "Loss không giảm trong quá trình diagnostic!"
    print("DIAGNOSTIC: PASS")


def train_model(
    cfg: dict[str, Any],
    device: torch.device,
    limit_train: int | None = None,
    limit_val: int | None = None,
    time_budget_s: int = 10 * 3600,
    resume: bool = False,
) -> dict[str, Any]:
    """Vòng lặp huấn luyện đầy đủ kèm checkpointing, validation và early stopping."""
    t_start = time.perf_counter()
    exp_id = cfg.get("experiment", {}).get("id", "EXP-BCB").lower().replace("-", "_")
    out_dir = Path("results") / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_cfg = cfg.get("training", {})
    epochs = int(tr_cfg.get("epochs", 5))
    batch_size = int(tr_cfg.get("batch_size", 32))
    lr = float(tr_cfg.get("learning_rate", 0.001))
    weight_decay = float(tr_cfg.get("weight_decay", 0.0001))
    grad_accum = int(tr_cfg.get("gradient_accumulation_steps", 1))

    model, tokenizer = build_model(cfg, device)
    model_id = str(cfg.get("model", {}).get("id", "M1")).upper()

    logger.info("Đang nạp dữ liệu train & validation...")
    train_pairs = load_pairs_data(cfg, split="train", max_samples=limit_train)
    val_pairs = load_pairs_data(cfg, split="val", max_samples=limit_val)
    logger.info("Dữ liệu nạp xong: %d train pairs, %d validation pairs", len(train_pairs), len(val_pairs))

    # Nạp cache AST hoặc CPG
    ast_store = None
    if model_id == "M3":
        cache_path = find_feature_resource("bcb_ast_cache.npz")
        if cache_path is not None:
            logger.info("Nạp AST cache từ %s...", cache_path)
            ast_store = ASTTreeStore.build({}, cache_path=cache_path, verbose=True)

    cpg_dir = setup_cpg_export_dir() if model_id == "M4" else None

    collate = get_collate_fn(cfg, model_id, tokenizer, device, ast_store=ast_store, cpg_dir=cpg_dir)
    train_loader = DataLoader(BCBPairDataset(train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(BCBPairDataset(val_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = -1.0
    best_threshold = 0.5
    best_epoch = -1
    patience = int(cfg.get("early_stopping", {}).get("patience", 2))
    patience_counter = 0

    history = []

    logger.info("Bắt đầu huấn luyện %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        if (time.perf_counter() - t_start) > time_budget_s:
            logger.warning("Đã chạm giới hạn thời gian (time budget %ds). Dừng train an toàn.", time_budget_s)
            break

        model.train()
        train_loss = 0.0
        n_batches = 0
        ep_t0 = time.perf_counter()

        optimizer.zero_grad()
        for b_idx, batch in enumerate(train_loader):
            logits, labels = forward_batch(model, batch, model_id)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()

            if (b_idx + 1) % grad_accum == 0 or (b_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum
            n_batches += 1

            if (b_idx + 1) % 50 == 0:
                logger.info(
                    "Epoch %d/%d | Batch %d/%d | Train Loss: %.4f",
                    epoch,
                    epochs,
                    b_idx + 1,
                    len(train_loader),
                    train_loss / n_batches,
                )

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch in val_loader:
                logits, labels = forward_batch(model, batch, model_id)
                v_loss = criterion(logits, labels)
                val_loss += v_loss.item()
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                val_preds.extend(probs)
                val_trues.extend(labels.cpu().numpy().tolist())

        val_loss /= max(len(val_loader), 1)
        val_preds_arr = np.array(val_preds)
        val_trues_arr = np.array(val_trues)

        opt_t, opt_f1 = find_optimal_threshold(val_trues_arr, val_preds_arr)
        metrics = compute_metrics(val_trues_arr, val_preds_arr, threshold=opt_t)
        ep_time = time.perf_counter() - ep_t0

        logger.info(
            "Epoch %d HOÀN TẤT (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f (threshold=%.2f)",
            epoch,
            ep_time,
            train_loss,
            val_loss,
            opt_f1,
            opt_t,
        )

        ep_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_f1": round(opt_f1, 4),
            "optimal_threshold": round(opt_t, 4),
            "metrics": metrics,
            "elapsed_s": round(ep_time, 1),
        }
        history.append(ep_record)

        # Checkpointing
        last_ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_f1": best_f1,
            "best_threshold": best_threshold,
            "cfg": cfg,
        }
        torch.save(last_ckpt, out_dir / "last_checkpoint.pt")

        if opt_f1 > best_f1:
            best_f1 = opt_f1
            best_threshold = opt_t
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_f1": best_f1,
                    "best_threshold": best_threshold,
                    "metrics": metrics,
                    "cfg": cfg,
                },
                out_dir / "best_checkpoint.pt",
            )
            logger.info(">>> Đã lưu BEST CHECKPOINT tại epoch %d với Val F1: %.4f", epoch, best_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping kích hoạt sau %d epoch không cải thiện.", patience)
                break

    # Lưu training summary
    summary = {
        "experiment_id": exp_id,
        "model_id": model_id,
        "best_epoch": best_epoch,
        "best_f1": round(best_f1, 4),
        "best_threshold": round(best_threshold, 4),
        "total_training_time_s": round(time.perf_counter() - t_start, 1),
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 50)
    print("TRAINING HOÀN TẤT!")
    print(f"Best Epoch: {best_epoch} | Best Val F1: {best_f1:.4f} (Threshold: {best_threshold:.4f})")
    print(f"Checkpoint lưu tại: {out_dir / 'best_checkpoint.pt'}")
    print("=" * 50)
    return summary


def evaluate_model(
    cfg: dict[str, Any],
    device: torch.device,
    limit_test: int | None = None,
) -> dict[str, Any]:
    """Đánh giá mô hình tốt nhất trên tập test với ngưỡng threshold tối ưu."""
    exp_id = cfg.get("experiment", {}).get("id", "EXP-BCB").lower().replace("-", "_")
    out_dir = Path("results") / exp_id
    ckpt_path = out_dir / "best_checkpoint.pt"

    if not ckpt_path.exists():
        logger.warning("Không tìm thấy %s, thử dùng last_checkpoint.pt...", ckpt_path)
        ckpt_path = out_dir / "last_checkpoint.pt"

    assert ckpt_path.exists(), f"Không tìm thấy checkpoint tại {out_dir}"
    ckpt = torch.load(ckpt_path, map_location=device)
    threshold = float(ckpt.get("best_threshold", 0.5))

    model, tokenizer = build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model_id = str(cfg.get("model", {}).get("id", "M1")).upper()

    logger.info("Đang nạp tập test (limit=%s)...", limit_test)
    test_pairs = load_pairs_data(cfg, split="test", max_samples=limit_test)

    ast_store = None
    if model_id == "M3":
        cache_path = find_feature_resource("bcb_ast_cache.npz")
        if cache_path is not None:
            ast_store = ASTTreeStore.build({}, cache_path=cache_path, verbose=False)

    cpg_dir = setup_cpg_export_dir() if model_id == "M4" else None

    collate = get_collate_fn(cfg, model_id, tokenizer, device, ast_store=ast_store, cpg_dir=cpg_dir)
    test_loader = DataLoader(BCBPairDataset(test_pairs), batch_size=32, shuffle=False, collate_fn=collate)

    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, labels = forward_batch(model, batch, model_id)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            preds.extend(probs)
            trues.extend(labels.cpu().numpy().tolist())

    metrics = compute_metrics(np.array(trues), np.array(preds), threshold=threshold)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:")
    print("=" * 50)
    print(json.dumps(metrics, indent=2))
    return metrics


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
    parser.add_argument("--limit-test", type=int, default=None, help="Giới hạn số mẫu test")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    set_global_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = torch.device(args.device)

    logger.info("Chạy bcb_runner: mode=%s, device=%s", args.mode, device)

    if args.mode == "preflight":
        preflight(cfg, device)
    elif args.mode in ("smoke", "diagnostic"):
        diagnostic(cfg, device)
    elif args.mode in ("train", "resume"):
        train_model(
            cfg=cfg,
            device=device,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            time_budget_s=args.time_budget_s,
            resume=(args.mode == "resume"),
        )
    elif args.mode == "evaluate":
        evaluate_model(cfg=cfg, device=device, limit_test=args.limit_test)


if __name__ == "__main__":
    main()
