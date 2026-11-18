"""Huấn luyện Cross-Encoder cho M1 (CodeBERT) và M2 (GraphCodeBERT) trên BigCloneBench 30k.

Cấu trúc Cross-Encoder:
  Input: <s> func1 </s></s> func2 </s> (512 tokens)
  Self-attention 12 tầng cho phép các token của func1 và func2 tương tác trực tiếp (Cross-Interaction).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cross_encoders")


# =========================================================================
# UTILITIES: SEED & RESOURCE LOCATOR
# =========================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_file(filename: str) -> Path | None:
    search_paths = [
        Path(f"results/{filename}"),
        Path(f"data/{filename}"),
        Path(f"data/raw/bigclonebench/{filename}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/{filename}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/raw/bigclonebench/{filename}"),
        Path(f"/kaggle/working/{filename}"),
        Path(f"/kaggle/working/data/{filename}"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    for root in [Path("results"), Path("data"), Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for sub in root.rglob(filename):
                if sub.exists() and sub.is_file():
                    return sub
    return None


# =========================================================================
# DATASET & TOKENIZATION
# =========================================================================
class CodePairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        return {
            "func1": str(p["func1"]),
            "func2": str(p["func2"]),
            "label": float(p["label"]),
        }


def make_collate_fn(tokenizer: Any, max_length: int = 512, device: torch.device = torch.device("cpu")):
    def collate_fn(batch: List[Dict[str, Any]]):
        f1 = [b["func1"] for b in batch]
        f2 = [b["func2"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32, device=device)
        enc = tokenizer(
            f1,
            f2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, labels
    return collate_fn


# =========================================================================
# MODEL ARCHITECTURE: CROSS-ENCODER
# =========================================================================
class TransformerCrossEncoder(nn.Module):
    def __init__(self, hf_id: str, dropout: float = 0.2):
        super().__init__()
        self.hf_id = hf_id
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Using [CLS] token (position 0)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_rep).squeeze(-1)
        return logits


# =========================================================================
# EVALUATION & METRICS
# =========================================================================
def evaluate_probs(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for enc, labels in loader:
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(**enc)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_f1 = -1.0
    best_tau = 0.5
    best_metrics = {}

    for tau in np.linspace(0.05, 0.95, 91):
        preds = (probs >= tau).astype(int)
        y = labels.astype(int)
        f1 = float(f1_score(y, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(round(tau, 2))
            acc = float(accuracy_score(y, preds))
            prec = float(precision_score(y, preds, zero_division=0))
            rec = float(recall_score(y, preds, zero_division=0))
            tp = int(((preds == 1) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            try:
                auc = float(roc_auc_score(y, probs))
            except Exception:
                auc = 0.0

            best_metrics = {
                "threshold": best_tau,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
            }
    return best_tau, best_metrics


# =========================================================================
# TRAINING LOOP
# =========================================================================
def train_single_cross_encoder(
    model_name: str,
    hf_id: str,
    train_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    test_pairs: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Path,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 2e-5,
    weight_decay: float = 0.05,
    max_length: int = 512,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(">>> BẮT ĐẦU HUẤN LUYỆN CROSS-ENCODER: %s (%s) <<<", model_name, hf_id)
    logger.info("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    collate_fn = make_collate_fn(tokenizer, max_length=max_length, device=device)

    train_loader = DataLoader(CodePairDataset(train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(CodePairDataset(val_pairs), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(CodePairDataset(test_pairs), batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    model = TransformerCrossEncoder(hf_id=hf_id, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_f1 = -1.0
    best_metrics = {}
    history = []
    checkpoint_path = output_dir / f"best_cross_{model_name.lower()}.pt"

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        ep_t0 = time.perf_counter()

        for step, (enc, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(**enc)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if step % 100 == 0 or step == len(train_loader):
                logger.info(
                    "[%s] Epoch %d/%d | Step %d/%d | Train Loss: %.4f | LR: %.2e",
                    model_name,
                    epoch,
                    epochs,
                    step,
                    len(train_loader),
                    train_loss / n_batches,
                    scheduler.get_last_lr()[0],
                )

        ep_duration = time.perf_counter() - ep_t0
        avg_train_loss = train_loss / n_batches

        # Validation
        val_probs, val_labels = evaluate_probs(model, val_loader)
        tau_val, val_metrics = find_best_threshold(val_probs, val_labels)

        logger.info(
            "[%s] Epoch %d Kết thúc (%.1fs) | Loss: %.4f | Val F1: %.4f (tau=%.2f) | Val Acc: %.4f | Val AUC: %.4f",
            model_name,
            epoch,
            ep_duration,
            avg_train_loss,
            val_metrics["f1"],
            tau_val,
            val_metrics["accuracy"],
            val_metrics["roc_auc"],
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "time_sec": ep_duration,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(">>> Đã lưu Best Checkpoint tại Epoch %d (Val F1: %.4f) <<<", epoch, best_val_f1)

    total_time_h = (time.perf_counter() - t_start) / 3600.0
    logger.info("Huấn luyện hoàn thành sau %.2f giờ.", total_time_h)

    # Nạp lại checkpoint tốt nhất để đánh giá trên 5.000 test set
    logger.info(">>> Đang đánh giá trên tập kiểm thử chuẩn 5.000 mẫu BigCloneBench...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_probs, test_labels = evaluate_probs(model, test_loader)
    best_tau, test_metrics = find_best_threshold(test_probs, test_labels)
    test_metrics["train_time_hours"] = round(total_time_h, 2)
    test_metrics["model_name"] = model_name
    test_metrics["hf_id"] = hf_id

    logger.info("=" * 80)
    logger.info("[%s TEST RESULTS] tau=%.2f | F1=%.4f | Acc=%.4f | Prec=%.4f | Rec=%.4f | AUC=%.4f",
                model_name, best_tau, test_metrics["f1"], test_metrics["accuracy"],
                test_metrics["precision"], test_metrics["recall"], test_metrics["roc_auc"])
    logger.info("[%s CONFUSION MATRIX] TP=%d | TN=%d | FP=%d | FN=%d",
                model_name, test_metrics["tp"], test_metrics["tn"], test_metrics["fp"], test_metrics["fn"])
    logger.info("=" * 80)

    # Lưu kết quả
    probs_file = output_dir / f"cross_{model_name.lower()}_test_probs.npy"
    np.save(probs_file, test_probs)

    metrics_file = output_dir / f"cross_{model_name.lower()}_test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    history_file = output_dir / f"cross_{model_name.lower()}_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return test_metrics


# =========================================================================
# DATA LOADER
# =========================================================================
def load_all_datasets(n_train: int = 30000, n_val: int = 2000) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # 1. Load 5k test set
    test_file = locate_file("bcb_5k_test_pairs.jsonl")
    if not test_file:
        raise FileNotFoundError("Không tìm thấy bcb_5k_test_pairs.jsonl!")
    test_pairs = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line))
    logger.info("Đã nạp %d cặp test chuẩn từ %s", len(test_pairs), test_file)

    # 2. Load train from parquet
    train_file = locate_file("train_shard_0.parquet")
    if not train_file:
        raise FileNotFoundError("Không tìm thấy train_shard_0.parquet!")
    df_train = pd.read_parquet(train_file)
    half_train = n_train // 2
    df_pos = df_train[df_train["label"] == True].sample(n=half_train, random_state=42)
    df_neg = df_train[df_train["label"] == False].sample(n=half_train, random_state=42)
    df_sampled = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_pairs = df_sampled[["func1", "func2", "label"]].to_dict(orient="records")
    for p in train_pairs:
        p["label"] = 1 if p["label"] else 0
    logger.info("Đã nạp & cân bằng %d cặp train (15k clone, 15k non-clone)", len(train_pairs))

    # 3. Load val from parquet
    val_file = locate_file("validation_shard_0.parquet")
    if not val_file:
        raise FileNotFoundError("Không tìm thấy validation_shard_0.parquet!")
    df_val = pd.read_parquet(val_file)
    half_val = n_val // 2
    df_vpos = df_val[df_val["label"] == True].sample(n=min(half_val, len(df_val[df_val["label"] == True])), random_state=42)
    df_vneg = df_val[df_val["label"] == False].sample(n=min(half_val, len(df_val[df_val["label"] == False])), random_state=42)
    df_v_sampled = pd.concat([df_vpos, df_vneg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_pairs = df_v_sampled[["func1", "func2", "label"]].to_dict(orient="records")
    for p in val_pairs:
        p["label"] = 1 if p["label"] else 0
    logger.info("Đã nạp & cân bằng %d cặp validation (1k clone, 1k non-clone)", len(val_pairs))

    return train_pairs, val_pairs, test_pairs


# =========================================================================
# MAIN DRIVER
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Cross-Encoders M1 & M2")
    parser.add_argument("--model", type=str, choices=["both", "m1", "m2"], default="both")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="results/exp_cross_encoders")
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Thiết bị tính toán: %s (CUDA Available: %s)", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU Model: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / (1024**3))

    train_pairs, val_pairs, test_pairs = load_all_datasets()

    all_results = {}

    if args.model in ("both", "m1"):
        res_m1 = train_single_cross_encoder(
            model_name="M1_CodeBERT_Cross",
            hf_id="microsoft/codebert-base",
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            device=device,
            output_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
        )
        all_results["M1_CodeBERT_Cross"] = res_m1

    if args.model in ("both", "m2"):
        res_m2 = train_single_cross_encoder(
            model_name="M2_GraphCodeBERT_Cross",
            hf_id="microsoft/graphcodebert-base",
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            device=device,
            output_dir=out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
        )
        all_results["M2_GraphCodeBERT_Cross"] = res_m2

    # Lưu tổng hợp
    summary_path = out_dir / "all_cross_encoders_test_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(">>> ĐÃ LƯU TOÀN BỘ KẾT QUẢ TẠI %s <<<", summary_path)

    # Đóng gói tar.gz
    bundle_path = out_dir / "bcb_cross_encoders_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for f in out_dir.iterdir():
            if f.name != bundle_path.name:
                tar.add(f, arcname=f.name)
    logger.info(">>> ĐÃ ĐÓNG GÓI BUNDLE: %s (%.2f MB) <<<", bundle_path, bundle_path.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
