"""Kịch bản Huấn luyện và Đánh giá Cấp độ 3 (End-to-End Deep Graph-Transformer M5).

Huấn luyện liên kết sâu hai chiều giữa GraphCodeBERT và CPG-GNN trên tập dữ liệu BigCloneBench Java.
Tự động định vị tài nguyên trên môi trường Local hoặc Kaggle GPU (T4/P100).
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
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Thêm PROJECT_ROOT vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hybrid.level3 import DeepGraphTransformerPairClassifier
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.ingest import parse_cpg_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_hybrid_level3")


class HybridPairDataset(Dataset):
    """Dataset chứa các cặp hàm mã nguồn kèm ID đồ thị CPG."""

    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.pairs[idx]


def build_cpg_cache(pair_lists: List[List[Dict[str, Any]]], cpg_dir: Path) -> Dict[str, CPGraph]:
    """Tải trước toàn bộ các đồ thị CPG duy nhất của tập dữ liệu vào RAM."""
    unique_ids = set()
    for plist in pair_lists:
        for p in plist:
            if p.get("id1"):
                unique_ids.add(str(p["id1"]))
            if p.get("id2"):
                unique_ids.add(str(p["id2"]))

    logger.info("Tìm thấy %d mã hàm CPG duy nhất cần nạp vào RAM...", len(unique_ids))
    cache: Dict[str, CPGraph] = {}
    missing = 0
    t0 = time.perf_counter()

    for sid in unique_ids:
        json_file = cpg_dir / f"{sid}.json"
        if json_file.exists():
            try:
                g, _ = parse_cpg_file(json_file, max_nodes=1500)
                cache[sid] = g
            except Exception as e:
                missing += 1
        else:
            missing += 1

    t1 = time.perf_counter()
    logger.info(
        "Đã nạp thành công %d/%d đồ thị CPG vào RAM trong %.2f giây (thiếu: %d)",
        len(cache),
        len(unique_ids),
        t1 - t0,
        missing,
    )
    return cache


def create_collate_fn(tokenizer: Any, cpg_cache: Dict[str, CPGraph], max_length: int = 384):
    """Hàm collate tối ưu: tokenize văn bản và ghép batch PyG trực tiếp từ RAM."""

    dummy_graph = CPGraph(
        node_type=np.zeros(1, dtype=np.int64),
        node_feat=np.zeros((1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros(0, dtype=np.int64),
    )

    def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], Any, Dict[str, torch.Tensor], Any, torch.Tensor]:
        c1 = [p["func1"] for p in batch]
        c2 = [p["func2"] for p in batch]
        labels = [float(p["label"]) for p in batch]

        inputs_a = tokenizer(c1, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs_b = tokenizer(c2, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        graphs_a = [cpg_cache.get(str(p.get("id1")), dummy_graph) for p in batch]
        graphs_b = [cpg_cache.get(str(p.get("id2")), dummy_graph) for p in batch]

        pyg_a = collate_pyg(graphs_a)
        pyg_b = collate_pyg(graphs_b)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        return inputs_a, pyg_a, inputs_b, pyg_b, label_tensor

    return collate_fn


def find_file_in_paths(candidates: List[Path]) -> Path | None:
    for p in candidates:
        if p and p.exists():
            return p
    return None


def locate_cpg_dir() -> Path:
    candidates = [
        Path("artifacts/cpg_export"),
        Path("/kaggle/input/semanticcodeclonedetection-dataset-m4/bcb_cpg_export/cpg_export"),
        Path("/kaggle/input/semanticcodeclonedetection-dataset-m4/cpg_export"),
        Path("/kaggle/working/artifacts/cpg_export"),
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.json")):
            return p
    # Tìm kiếm đệ quy
    for root in [Path("artifacts"), Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for sub in root.rglob("cpg_export"):
                if sub.is_dir() and any(sub.glob("*.json")):
                    return sub
    return Path("artifacts/cpg_export")


def find_checkpoint(candidates: List[Path]) -> Path | None:
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    all_probs: List[float] = []
    all_targets: List[int] = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    n_batches = 0

    with torch.no_grad():
        for inputs_a, pyg_a, inputs_b, pyg_b, labels in loader:
            inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
            inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
            pyg_a = pyg_a.to(device)
            pyg_b = pyg_b.to(device)
            labels = labels.to(device)

            with autocast():
                logits = model(inputs_a, pyg_a, inputs_b, pyg_b)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy().astype(int).tolist())

    all_probs_np = np.array(all_probs)
    all_targets_np = np.array(all_targets)
    all_preds_np = (all_probs_np >= threshold).astype(int)

    acc = float(accuracy_score(all_targets_np, all_preds_np))
    prec = float(precision_score(all_targets_np, all_preds_np, zero_division=0))
    rec = float(recall_score(all_targets_np, all_preds_np, zero_division=0))
    f1 = float(f1_score(all_targets_np, all_preds_np, zero_division=0))
    try:
        auc = float(roc_auc_score(all_targets_np, all_probs_np))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_targets_np, all_preds_np)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
        "probs": all_probs_np,
        "targets": all_targets_np,
    }


def find_best_threshold(probs: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.20, 0.81, 0.02):
        preds = (probs >= th).astype(int)
        score = f1_score(targets, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_th = float(th)
    return best_th, best_f1


def main():
    parser = argparse.ArgumentParser(description="Train M5 Level 3 Deep Graph-Transformer")
    parser.add_argument("--epochs", type=int, default=6, help="Số epochs huấn luyện")
    parser.add_argument("--batch-size", type=int, default=16, help="Kích thước batch trên GPU")
    parser.add_argument("--grad-accum", type=int, default=2, help="Số bước tích lũy gradient (effective batch size = 32)")
    parser.add_argument("--lr-text", type=float, default=1.5e-5, help="Learning rate cho GraphCodeBERT")
    parser.add_argument("--lr-graph", type=float, default=1e-4, help="Learning rate cho CPG-GNN")
    parser.add_argument("--lr-head", type=float, default=2e-4, help="Learning rate cho Fusion & Classifier Head")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Trọng số suy giảm AdamW")
    parser.add_argument("--max-length", type=int, default=384, help="Độ dài token tối đa cho mã nguồn")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/exp_m5_level3")
    parser.add_argument("--m2-ckpt", type=str, default=None, help="Đường dẫn checkpoint warm-start M2-DR")
    parser.add_argument("--m4-ckpt", type=str, default=None, help="Đường dẫn checkpoint warm-start M4-EP")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Thiết lập seed ngẫu nhiên
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 75)
    print(">>> HUẤN LUYỆN MÔ HÌNH LAI GHÉP CẤP ĐỘ 3: DEEP GRAPH-TRANSFORMER (M5) <<<")
    print(f"Thiết bị tính toán: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"Cấu hình: Epochs={args.epochs}, BatchSize={args.batch_size}, GradAccum={args.grad_accum} (Effective={args.batch_size*args.grad_accum})")
    print(f"Learning Rates: Text={args.lr_text:.1e}, Graph={args.lr_graph:.1e}, Head={args.lr_head:.1e}")
    print("=" * 75)

    # 1. Dò tìm và định vị dữ liệu
    cpg_dir = locate_cpg_dir()
    logger.info("Thư mục CPG Graphs: %s (tổng cộng %d files)", cpg_dir, len(list(cpg_dir.glob("*.json"))))

    train_path = find_file_in_paths([
        Path("results/bcb_hybrid_train_6k.jsonl"),
        Path("/kaggle/working/project/results/bcb_hybrid_train_6k.jsonl"),
        Path("data/bcb_hybrid_train_6k.jsonl"),
    ])
    val_path = find_file_in_paths([
        Path("results/bcb_hybrid_val_2k.jsonl"),
        Path("/kaggle/working/project/results/bcb_hybrid_val_2k.jsonl"),
        Path("data/bcb_hybrid_val_2k.jsonl"),
    ])
    test_path = find_file_in_paths([
        Path("results/bcb_5k_test_pairs.jsonl"),
        Path("/kaggle/working/project/results/bcb_5k_test_pairs.jsonl"),
        Path("data/bcb_5k_test_pairs.jsonl"),
    ])

    if not train_path or not val_path or not test_path:
        raise FileNotFoundError(f"Không tìm đủ file dữ liệu: train={train_path}, val={val_path}, test={test_path}")

    # Đọc dữ liệu JSONL
    def read_jsonl(p: Path) -> List[Dict[str, Any]]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train_pairs = read_jsonl(train_path)
    val_pairs = read_jsonl(val_path)
    test_pairs = read_jsonl(test_path)
    logger.info("Dữ liệu nạp vào: Train=%d, Val=%d, Test=%d cặp", len(train_pairs), len(val_pairs), len(test_pairs))

    # 2. Tải trước CPG Graphs vào RAM
    cpg_cache = build_cpg_cache([train_pairs, val_pairs, test_pairs], cpg_dir)

    # 3. Khởi tạo Tokenizer và DataLoaders
    hf_id = "microsoft/graphcodebert-base"
    logger.info("Khởi tạo tokenizer: %s (max_length=%d)...", hf_id, args.max_length)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    collate = create_collate_fn(tokenizer, cpg_cache, max_length=args.max_length)

    train_loader = DataLoader(
        HybridPairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        HybridPairDataset(val_pairs),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        HybridPairDataset(test_pairs),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # 4. Khởi tạo Mô hình DeepGraphTransformerPairClassifier
    logger.info("Xây dựng mô hình End-to-End Deep Graph-Transformer...")
    model = DeepGraphTransformerPairClassifier(
        hf_id=hf_id,
        graph_hidden=128,
        graph_num_layers=3,
        graph_heads=4,
        graph_dropout=0.2,
        head_hidden=512,
        head_dropout=0.2,
        freeze_text=False,
        freeze_graph=False,
    ).to(device)

    counts = model.param_counts()
    logger.info(
        "Tổng tham số: %s | Huấn luyện: %s (Text: %s, Graph: %s, Fusion+Head: %s)",
        f"{counts['total']:,}",
        f"{counts['trainable']:,}",
        f"{counts['text']:,}",
        f"{counts['graph']:,}",
        f"{counts['fusion_head']:,}",
    )

    # 5. Nạp warm-start từ M2-DR và M4-EP nếu có
    m2_ckpt = args.m2_ckpt or find_checkpoint([
        Path("results/exp_m2_dr/best_checkpoint.pt"),
        Path("/kaggle/input/bcb-tuning-m2-dr/exp_m2_dr_best_checkpoint.pt"),
        Path("/kaggle/input/bcb-tuning-m2-dr/best_checkpoint.pt"),
    ])
    m4_ckpt = args.m4_ckpt or find_checkpoint([
        Path("results/exp_m4_ep/best_checkpoint.pt"),
        Path("/kaggle/input/bcb-tuning-m4-cpg-gnn/exp_m4_ep_best_checkpoint.pt"),
        Path("/kaggle/input/bcb-tuning-m4-cpg-gnn/best_checkpoint.pt"),
    ])

    warm_status = model.load_warm_start(m2_ckpt, m4_ckpt, device=device)
    logger.info("Kết quả Warm-start: Text(M2)=%s, Graph(M4)=%s", warm_status["text"], warm_status["graph"])

    # 6. Thiết lập Optimizer và Cosine Scheduler
    param_groups = model.get_optimizer_param_groups(
        lr_text=args.lr_text,
        lr_graph=args.lr_graph,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    total_training_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.BCEWithLogitsLoss()

    logger.info("Tổng số bước tối ưu hóa: %d (warmup=%d bước)", total_training_steps, warmup_steps)

    # 7. Vòng lặp huấn luyện chính
    best_val_f1 = -1.0
    best_val_epoch = -1
    best_threshold = 0.5
    history = []
    t_train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        t_epoch_start = time.perf_counter()
        optimizer.zero_grad()

        for step, (inputs_a, pyg_a, inputs_b, pyg_b, labels) in enumerate(train_loader, 1):
            inputs_a = {k: v.to(device) for k, v in inputs_a.items()}
            inputs_b = {k: v.to(device) for k, v in inputs_b.items()}
            pyg_a = pyg_a.to(device)
            pyg_b = pyg_b.to(device)
            labels = labels.to(device)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(inputs_a, pyg_a, inputs_b, pyg_b)
                loss = criterion(logits, labels) / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum
            n_steps += 1

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % 50 == 0 or step == len(train_loader):
                current_lr_text = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch}/{args.epochs}] Step {step}/{len(train_loader)} | "
                    f"Batch Loss: {running_loss / n_steps:.4f} | LR Text: {current_lr_text:.2e}",
                    flush=True,
                )

        train_loss = running_loss / max(n_steps, 1)
        epoch_time = time.perf_counter() - t_epoch_start

        # Đánh giá trên Validation
        val_res = evaluate_model(model, val_loader, device=device, threshold=0.5)
        best_th_val, best_f1_val = find_best_threshold(val_res["probs"], val_res["targets"])

        epoch_stat = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_res["loss"], 4),
            "val_acc": round(val_res["accuracy"] * 100, 2),
            "val_prec": round(val_res["precision"] * 100, 2),
            "val_rec": round(val_res["recall"] * 100, 2),
            "val_f1_05": round(val_res["f1"] * 100, 2),
            "val_f1_best": round(best_f1_val * 100, 2),
            "best_threshold": round(best_th_val, 2),
            "val_auc": round(val_res["roc_auc"] * 100, 2),
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(epoch_stat)

        print(
            f">>> Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_res['loss']:.4f} | "
            f"Val F1: {val_res['f1']*100:.2f}% (Tối ưu: {best_f1_val*100:.2f}% @ th={best_th_val:.2f}) | "
            f"Val Prec: {val_res['precision']*100:.2f}% | Val Rec: {val_res['recall']*100:.2f}%",
            flush=True,
        )

        # Lưu checkpoint tốt nhất theo val_f1
        if best_f1_val > best_val_f1:
            best_val_f1 = best_f1_val
            best_val_epoch = epoch
            best_threshold = best_th_val
            ckpt_data = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": epoch_stat,
                "best_threshold": best_threshold,
                "param_counts": counts,
            }
            torch.save(ckpt_data, output_dir / "best_checkpoint.pt")
            logger.info(">>> ĐÃ LƯU CHECKPOINT TỐT NHẤT (Val F1: %.2f%%) tại Epoch %d <<<", best_f1_val * 100, epoch)

        # Lưu last checkpoint
        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "val_metrics": epoch_stat},
            output_dir / "last_checkpoint.pt",
        )

    total_train_time = time.perf_counter() - t_train_start
    print(f"\n>>> HOÀN TẤT HUẤN LUYỆN trong {total_train_time/60:.2f} phút! <<<")
    print(f">>> Kỷ lục Validation: Epoch {best_val_epoch} với Val F1 = {best_val_f1*100:.2f}% (Ngưỡng: {best_threshold:.2f}) <<<")

    # 8. Đánh giá kiểm định cuối cùng trên 5.000 cặp Test
    print("\n" + "=" * 75)
    print(">>> TIẾN HÀNH ĐÁNH GIÁ TRÊN TẬP KIỂM ĐỊNH ĐỘC LẬP (5.000 CẶP TEST) <<<")
    print("=" * 75)

    # Nạp lại checkpoint tốt nhất
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    saved_th = best_ckpt.get("best_threshold", best_threshold)
    logger.info("Đã nạp checkpoint tốt nhất từ Epoch %d (ngưỡng tối ưu: %.2f)", best_ckpt["epoch"], saved_th)

    test_res_default = evaluate_model(model, test_loader, device=device, threshold=0.5)
    test_res_opt = evaluate_model(model, test_loader, device=device, threshold=saved_th)

    print(f"\n[Kết quả Test @ Ngưỡng mặc định 0.50]:")
    print(f"  - Accuracy : {test_res_default['accuracy']*100:.2f}%")
    print(f"  - Precision: {test_res_default['precision']*100:.2f}%")
    print(f"  - Recall   : {test_res_default['recall']*100:.2f}%")
    print(f"  - F1-Score : {test_res_default['f1']*100:.2f}%")
    print(f"  - ROC-AUC  : {test_res_default['roc_auc']*100:.2f}%")
    print(f"  - Confusion: TP={test_res_default['tp']}, FP={test_res_default['fp']}, FN={test_res_default['fn']}, TN={test_res_default['tn']}")

    print(f"\n[Kết quả Test @ Ngưỡng tối ưu từ Val {saved_th:.2f}]:")
    print(f"  - Accuracy : {test_res_opt['accuracy']*100:.2f}%")
    print(f"  - Precision: {test_res_opt['precision']*100:.2f}%")
    print(f"  - Recall   : {test_res_opt['recall']*100:.2f}%")
    print(f"  - F1-Score : {test_res_opt['f1']*100:.2f}%")
    print(f"  - ROC-AUC  : {test_res_opt['roc_auc']*100:.2f}%")
    print(f"  - Confusion: TP={test_res_opt['tp']}, FP={test_res_opt['fp']}, FN={test_res_opt['fn']}, TN={test_res_opt['tn']}")

    # Lưu xác suất dự đoán test để đối chiếu oracle sau này
    np.save(output_dir / "m5_level3_5k_probs.npy", test_res_opt["probs"])

    # Tổng kết và xuất file JSON
    final_metrics = {
        "model_id": "M5",
        "model_name": "Deep Graph-Transformer (GraphCodeBERT + CPG-GNN End-to-End)",
        "level": 3,
        "best_epoch": best_val_epoch,
        "best_threshold": saved_th,
        "train_time_seconds": round(total_train_time, 2),
        "train_time_hours": round(total_train_time / 3600, 2),
        "params": counts,
        "metrics_default_threshold": {
            "accuracy": round(test_res_default["accuracy"] * 100, 2),
            "precision": round(test_res_default["precision"] * 100, 2),
            "recall": round(test_res_default["recall"] * 100, 2),
            "f1": round(test_res_default["f1"] * 100, 2),
            "roc_auc": round(test_res_default["roc_auc"] * 100, 2),
            "tp": test_res_default["tp"],
            "fp": test_res_default["fp"],
            "fn": test_res_default["fn"],
            "tn": test_res_default["tn"],
        },
        "metrics_optimal_threshold": {
            "accuracy": round(test_res_opt["accuracy"] * 100, 2),
            "precision": round(test_res_opt["precision"] * 100, 2),
            "recall": round(test_res_opt["recall"] * 100, 2),
            "f1": round(test_res_opt["f1"] * 100, 2),
            "roc_auc": round(test_res_opt["roc_auc"] * 100, 2),
            "tp": test_res_opt["tp"],
            "fp": test_res_opt["fp"],
            "fn": test_res_opt["fn"],
            "tn": test_res_opt["tn"],
        },
    }

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "config": vars(args)}, f, indent=2, ensure_ascii=False)

    print(f"\n>>> Đã lưu toàn bộ kết quả vào thư mục: {output_dir}/ <<<")
    print("Hoàn tất quy trình huấn luyện và đánh giá Level 3 thành công!")


if __name__ == "__main__":
    main()
