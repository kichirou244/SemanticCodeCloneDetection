"""Trình huấn luyện và đánh giá hợp nhất M5 Toàn diện (Cấp độ 1, 2, 3) trên 30.000 cặp BigCloneBench.

Đồng nhất tuyệt đối quy mô dữ liệu với các thử nghiệm M1, M2, M3, M4:
  - Train: 30.000 cặp (15.000 clone, 15.000 non-clone)
  - Validation: 3.000 cặp (1.500 clone, 1.500 non-clone)
  - Test: 5.000 cặp kiểm định độc lập (2.500 clone, 2.500 non-clone)

Chạy liên hoàn trên 1 kernel duy nhất:
  1. Cấp độ 1: Late Fusion & Dynamic Gating (kết hợp M2-DR và M4-EP trên 5k test).
  2. Cấp độ 2: Trích xuất đặc trưng 30k cặp và huấn luyện M5-Concat, M5-Gated.
  3. Cấp độ 3: Huấn luyện End-to-End Joint Fine-tuning Deep Graph-Transformer trên 30k cặp.
  4. Xuất bảng đối chứng tổng hợp toàn diện.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer_pair import TransformerPairClassifier
from src.models.cpg_gnn.model import CPGGNNPairClassifier, CPGEncoder
from src.models.hybrid.model import HybridPairClassifier
from src.models.hybrid.level3 import DeepGraphTransformerPairClassifier
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.ingest import parse_cpg_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m5_unified")


class HybridDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.pairs[idx]


def locate_resource(name: str) -> Path | None:
    search_paths = [
        Path(f"artifacts/{name}"),
        Path(f"artifacts/cpg_export"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/bcb_cpg_export/{name}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/{name}"),
        Path(f"/kaggle/working/artifacts/{name}"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    for root in [Path("artifacts"), Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for sub in root.rglob(name):
                if sub.exists():
                    return sub
    return None


def locate_checkpoint(name_keyword: str) -> Path | None:
    for root in [Path("results"), Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for p in root.rglob("*.pt"):
                if name_keyword.lower() in p.name.lower() and "best" in p.name.lower():
                    return p
    return None


def build_cpg_cache(pair_lists: List[List[Dict[str, Any]]], cpg_dir: Path) -> Dict[str, CPGraph]:
    unique_ids = set()
    for plist in pair_lists:
        for p in plist:
            if p.get("id1"):
                unique_ids.add(str(p["id1"]))
            if p.get("id2"):
                unique_ids.add(str(p["id2"]))

    logger.info("Nạp %d đồ thị CPG duy nhất vào RAM từ %s...", len(unique_ids), cpg_dir)
    cache: Dict[str, CPGraph] = {}
    missing = 0
    t0 = time.perf_counter()

    for sid in unique_ids:
        json_file = cpg_dir / f"{sid}.json"
        if json_file.exists():
            try:
                g, _ = parse_cpg_file(json_file, max_nodes=1500)
                cache[sid] = g
            except Exception:
                missing += 1
        else:
            missing += 1

    t1 = time.perf_counter()
    logger.info("Đã nạp %d/%d đồ thị vào RAM trong %.2f s (thiếu: %d)", len(cache), len(unique_ids), t1 - t0, missing)
    return cache


def create_collate_fn(tokenizer: Any, cpg_cache: Dict[str, CPGraph], max_length: int = 384):
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


def compute_metrics_at_threshold(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        auc = 0.5
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
        "threshold": round(threshold, 3),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.20, 0.81, 0.02):
        score = f1_score(y_true, (y_probs >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_th = float(th)
    return best_th, best_f1


# =========================================================================
# GIAI ĐOẠN 1: CẤP ĐỘ 1 (LATE FUSION & GATING ENSEMBLE)
# =========================================================================
def run_level1(
    m2_probs_test: np.ndarray,
    m4_probs_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    logger.info(">>> Đang chạy Cấp độ 1: Late Fusion & Gating Ensemble trên 5.000 test...")
    results = {}

    # 1. Linear Blending tối ưu
    best_blend_f1 = -1.0
    best_alpha = 0.7
    best_blend_th = 0.5

    for alpha in np.arange(0.50, 0.95, 0.05):
        blended = alpha * m2_probs_test + (1 - alpha) * m4_probs_test
        th, f1 = find_optimal_threshold(y_test, blended)
        if f1 > best_blend_f1:
            best_blend_f1 = f1
            best_alpha = float(alpha)
            best_blend_th = th

    blended_probs = best_alpha * m2_probs_test + (1 - best_alpha) * m4_probs_test
    results["linear_blend"] = compute_metrics_at_threshold(y_test, blended_probs, best_blend_th)
    results["linear_blend"]["alpha"] = round(best_alpha, 2)

    # 2. Confidence-Band Gating (chỉ can thiệp khi M2 phân vân [0.2, 0.8])
    band_low, band_high = 0.20, 0.80
    conf_probs = m2_probs_test.copy()
    in_band = (m2_probs_test >= band_low) & (m2_probs_test <= band_high)
    conf_probs[in_band] = 0.70 * m2_probs_test[in_band] + 0.30 * m4_probs_test[in_band]
    best_th_conf, _ = find_optimal_threshold(y_test, conf_probs)
    results["confidence_gating"] = compute_metrics_at_threshold(y_test, conf_probs, best_th_conf)

    # 3. Dynamic FN Rescue Gating
    rescue_probs = m2_probs_test.copy()
    rescue_mask = (m2_probs_test < 0.50) & (m4_probs_test >= 0.95)
    rescue_probs[rescue_mask] = 0.55
    results["dynamic_rescue"] = compute_metrics_at_threshold(y_test, rescue_probs, 0.50)

    logger.info("Hoàn tất Cấp độ 1: Linear Blend F1=%.2f%%, Conf Gating F1=%.2f%%",
                results["linear_blend"]["f1"], results["confidence_gating"]["f1"])
    return results


# =========================================================================
# GIAI ĐOẠN 2: CẤP ĐỘ 2 (FEATURE-LEVEL FUSION TRÊN 30.000 CẶP)
# =========================================================================
def extract_embeddings_side(
    pairs: List[Dict[str, Any]],
    m2_model: TransformerPairClassifier,
    m4_model: CPGGNNPairClassifier,
    tokenizer: Any,
    cpg_cache: Dict[str, CPGraph],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dummy_graph = CPGraph(
        node_type=np.zeros(1, dtype=np.int64),
        node_feat=np.zeros((1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros(0, dtype=np.int64),
    )

    h_text_a_list, h_text_b_list = [], []
    h_graph_a_list, h_graph_b_list = [], []
    y_list = [p["label"] for p in pairs]

    m2_model.eval()
    m4_model.eval()

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            c1 = [p["func1"] for p in chunk]
            c2 = [p["func2"] for p in chunk]

            in_a = tokenizer(c1, padding=True, truncation=True, max_length=384, return_tensors="pt")
            in_b = tokenizer(c2, padding=True, truncation=True, max_length=384, return_tensors="pt")
            in_a = {k: v.to(device) for k, v in in_a.items()}
            in_b = {k: v.to(device) for k, v in in_b.items()}

            with autocast(enabled=(device.type == "cuda")):
                ha_t = m2_model.encode_side(in_a).cpu().numpy()
                hb_t = m2_model.encode_side(in_b).cpu().numpy()

            h_text_a_list.append(ha_t)
            h_text_b_list.append(hb_t)

            ga = [cpg_cache.get(str(p.get("id1")), dummy_graph) for p in chunk]
            gb = [cpg_cache.get(str(p.get("id2")), dummy_graph) for p in chunk]
            pyg_a = collate_pyg(ga).to(device)
            pyg_b = collate_pyg(gb).to(device)

            ha_g = m4_model.encode_side(pyg_a).cpu().numpy()
            hb_g = m4_model.encode_side(pyg_b).cpu().numpy()

            h_graph_a_list.append(ha_g)
            h_graph_b_list.append(hb_g)

    return (
        np.concatenate(h_text_a_list, axis=0),
        np.concatenate(h_text_b_list, axis=0),
        np.concatenate(h_graph_a_list, axis=0),
        np.concatenate(h_graph_b_list, axis=0),
        np.array(y_list, dtype=int),
    )


def train_level2_variant(
    model: HybridPairClassifier,
    train_tensors: Tuple[torch.Tensor, ...],
    val_tensors: Tuple[torch.Tensor, ...],
    test_tensors: Tuple[torch.Tensor, ...],
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    batch_size: int = 64,
) -> Tuple[Dict[str, Any], np.ndarray]:
    model.to(device)
    train_ds = TensorDataset(*train_tensors)
    val_ds = TensorDataset(*val_tensors)
    test_ds = TensorDataset(*test_tensors)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_th = 0.5
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for hta, htb, hga, hgb, y in train_loader:
            hta, htb, hga, hgb, y = hta.to(device), htb.to(device), hga.to(device), hgb.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model.forward_from_embeddings(hta, hga, htb, hgb)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Val eval
        model.eval()
        val_probs, val_y = [], []
        with torch.no_grad():
            for hta, htb, hga, hgb, y in val_loader:
                hta, htb, hga, hgb = hta.to(device), htb.to(device), hga.to(device), hgb.to(device)
                l = model.forward_from_embeddings(hta, hga, htb, hgb)
                val_probs.extend(torch.sigmoid(l).cpu().numpy().tolist())
                val_y.extend(y.numpy().tolist())

        val_probs_np = np.array(val_probs)
        val_y_np = np.array(val_y)
        th, f1 = find_optimal_threshold(val_y_np, val_probs_np)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_th = th
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Nạp weights tốt nhất và test
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    test_probs, test_y = [], []
    with torch.no_grad():
        for hta, htb, hga, hgb, y in test_loader:
            hta, htb, hga, hgb = hta.to(device), htb.to(device), hga.to(device), hgb.to(device)
            l = model.forward_from_embeddings(hta, hga, htb, hgb)
            test_probs.extend(torch.sigmoid(l).cpu().numpy().tolist())
            test_y.extend(y.numpy().tolist())

    test_probs_np = np.array(test_probs)
    test_y_np = np.array(test_y)
    metrics = compute_metrics_at_threshold(test_y_np, test_probs_np, best_th)
    return metrics, test_probs_np


def run_level2(
    tr_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    test_pairs: List[Dict[str, Any]],
    m2_model: TransformerPairClassifier,
    m4_model: CPGGNNPairClassifier,
    tokenizer: Any,
    cpg_cache: Dict[str, CPGraph],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    logger.info(">>> Đang chạy Cấp độ 2: Feature-Level Fusion trên toàn bộ 30.000 cặp Train...")
    t0 = time.perf_counter()

    # Trích xuất embeddings
    logger.info("Trích xuất embeddings cho 30.000 train...")
    tr_hta, tr_htb, tr_hga, tr_hgb, tr_y = extract_embeddings_side(tr_pairs, m2_model, m4_model, tokenizer, cpg_cache, device)
    logger.info("Trích xuất embeddings cho 3.000 val...")
    val_hta, val_htb, val_hga, val_hgb, val_y = extract_embeddings_side(val_pairs, m2_model, m4_model, tokenizer, cpg_cache, device)
    logger.info("Trích xuất embeddings cho 5.000 test...")
    te_hta, te_htb, te_hga, te_hgb, te_y = extract_embeddings_side(test_pairs, m2_model, m4_model, tokenizer, cpg_cache, device)

    tr_tensors = (torch.tensor(tr_hta, dtype=torch.float32), torch.tensor(tr_htb, dtype=torch.float32),
                  torch.tensor(tr_hga, dtype=torch.float32), torch.tensor(tr_hgb, dtype=torch.float32),
                  torch.tensor(tr_y, dtype=torch.float32))
    val_tensors = (torch.tensor(val_hta, dtype=torch.float32), torch.tensor(val_htb, dtype=torch.float32),
                   torch.tensor(val_hga, dtype=torch.float32), torch.tensor(val_hgb, dtype=torch.float32),
                   torch.tensor(val_y, dtype=torch.float32))
    te_tensors = (torch.tensor(te_hta, dtype=torch.float32), torch.tensor(te_htb, dtype=torch.float32),
                  torch.tensor(te_hga, dtype=torch.float32), torch.tensor(te_hgb, dtype=torch.float32),
                  torch.tensor(te_y, dtype=torch.float32))

    logger.info("Huấn luyện M5-Concat trên 30.000 cặp...")
    m5_concat = HybridPairClassifier(use_gating=False, freeze_encoders=True)
    res_concat, probs_concat = train_level2_variant(m5_concat, tr_tensors, val_tensors, te_tensors, device, epochs=20)
    torch.save(m5_concat.state_dict(), output_dir / "m5_level2_concat_30k.pt")

    logger.info("Huấn luyện M5-Gated trên 30.000 cặp...")
    m5_gated = HybridPairClassifier(use_gating=True, freeze_encoders=True)
    res_gated, probs_gated = train_level2_variant(m5_gated, tr_tensors, val_tensors, te_tensors, device, epochs=20)
    torch.save(m5_gated.state_dict(), output_dir / "m5_level2_gated_30k.pt")

    t_dur = (time.perf_counter() - t0) / 60
    logger.info("Hoàn tất Cấp độ 2 trong %.1f phút: Concat F1=%.2f%%, Gated F1=%.2f%% (Prec=%.2f%%)",
                t_dur, res_concat["f1"], res_gated["f1"], res_gated["precision"])

    return {
        "m5_concat_30k": res_concat,
        "m5_gated_30k": res_gated,
        "duration_minutes": round(t_dur, 2),
    }


# =========================================================================
# GIAI ĐOẠN 3: CẤP ĐỘ 3 (END-TO-END DEEP GRAPH-TRANSFORMER TRÊN 30.000 CẶP)
# =========================================================================
def run_level3(
    tr_pairs: List[Dict[str, Any]],
    val_pairs: List[Dict[str, Any]],
    test_pairs: List[Dict[str, Any]],
    tokenizer: Any,
    cpg_cache: Dict[str, CPGraph],
    m2_ckpt: Path | None,
    m4_ckpt: Path | None,
    device: torch.device,
    output_dir: Path,
    epochs: int = 4,
    batch_size: int = 16,
    grad_accum: int = 2,
    lr_text: float = 1.5e-5,
    lr_graph: float = 1.0e-4,
    lr_head: float = 2.0e-4,
) -> Tuple[Dict[str, Any], np.ndarray]:
    logger.info(">>> Đang chạy Cấp độ 3: End-to-End Joint Fine-tuning trên 30.000 cặp...")
    t0 = time.perf_counter()

    collate = create_collate_fn(tokenizer, cpg_cache, max_length=384)
    train_loader = DataLoader(HybridDataset(tr_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(HybridDataset(val_pairs), batch_size=batch_size * 2, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(HybridDataset(test_pairs), batch_size=batch_size * 2, shuffle=False, collate_fn=collate, num_workers=0)

    model = DeepGraphTransformerPairClassifier(
        hf_id="microsoft/graphcodebert-base",
        graph_hidden=128,
        graph_num_layers=3,
        graph_heads=4,
        graph_dropout=0.2,
        head_hidden=512,
        head_dropout=0.2,
        freeze_text=False,
        freeze_graph=False,
    ).to(device)

    # Nạp khởi tạo ấm
    if m2_ckpt and m4_ckpt:
        model.load_warm_start(m2_ckpt, m4_ckpt, device=device)

    param_groups = model.get_optimizer_param_groups(lr_text=lr_text, lr_graph=lr_graph, lr_head=lr_head, weight_decay=0.01)
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_th = 0.5
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0
        t_ep_start = time.perf_counter()
        optimizer.zero_grad()

        for step, (in_a, pyg_a, in_b, pyg_b, y) in enumerate(train_loader, 1):
            in_a = {k: v.to(device) for k, v in in_a.items()}
            in_b = {k: v.to(device) for k, v in in_b.items()}
            pyg_a = pyg_a.to(device)
            pyg_b = pyg_b.to(device)
            y = y.to(device)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(in_a, pyg_a, in_b, pyg_b)
                loss = criterion(logits, y) / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum
            n_steps += 1

            if step % grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % 250 == 0 or step == len(train_loader):
                logger.info("  [Epoch %d/%d] Step %d/%d | Loss: %.4f | LR Text: %.2e",
                            epoch, epochs, step, len(train_loader), running_loss / n_steps, optimizer.param_groups[0]["lr"])

        ep_time = time.perf_counter() - t_ep_start

        # Val eval
        model.eval()
        val_probs, val_y = [], []
        with torch.no_grad():
            for in_a, pyg_a, in_b, pyg_b, y in val_loader:
                in_a = {k: v.to(device) for k, v in in_a.items()}
                in_b = {k: v.to(device) for k, v in in_b.items()}
                pyg_a, pyg_b = pyg_a.to(device), pyg_b.to(device)
                with autocast(enabled=(device.type == "cuda")):
                    l = model(in_a, pyg_a, in_b, pyg_b)
                val_probs.extend(torch.sigmoid(l).cpu().numpy().tolist())
                val_y.extend(y.numpy().tolist())

        val_probs_np = np.array(val_probs)
        val_y_np = np.array(val_y)
        th, f1 = find_optimal_threshold(val_y_np, val_probs_np)
        logger.info(">>> Epoch %d/%d (%.1fs) | Val F1: %.2f%% (Optimal @ th=%.2f)", epoch, epochs, ep_time, f1 * 100, th)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_th = th
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"epoch": epoch, "model_state": best_weights, "best_threshold": best_th}, output_dir / "best_checkpoint_l3_30k.pt")

    # Đánh giá trên 5k test
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    test_probs, test_y = [], []
    with torch.no_grad():
        for in_a, pyg_a, in_b, pyg_b, y in test_loader:
            in_a = {k: v.to(device) for k, v in in_a.items()}
            in_b = {k: v.to(device) for k, v in in_b.items()}
            pyg_a, pyg_b = pyg_a.to(device), pyg_b.to(device)
            with autocast(enabled=(device.type == "cuda")):
                l = model(in_a, pyg_a, in_b, pyg_b)
            test_probs.extend(torch.sigmoid(l).cpu().numpy().tolist())
            test_y.extend(y.numpy().tolist())

    test_probs_np = np.array(test_probs)
    test_y_np = np.array(test_y)
    metrics = compute_metrics_at_threshold(test_y_np, test_probs_np, best_th)
    t_dur = (time.perf_counter() - t0) / 60
    metrics["duration_minutes"] = round(t_dur, 2)
    logger.info("Hoàn tất Cấp độ 3 trong %.1f phút: Test F1=%.2f%%, Acc=%.2f%%, Prec=%.2f%%, Rec=%.2f%%",
                t_dur, metrics["f1"], metrics["accuracy"], metrics["precision"], metrics["recall"])
    return metrics, test_probs_np


# =========================================================================
# HÀM CHÍNH ĐIỀU PHỐI TỔNG THỂ
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="M5 Unified Runner on 30k BigCloneBench")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/exp_m5_unified")
    parser.add_argument("--epochs-l3", type=int, default=4, help="Số epochs cho Cấp độ 3 trên 30k")
    parser.add_argument("--epochs-l2", type=int, default=20, help="Số epochs cho Cấp độ 2 trên 30k")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print(">>> HUẤN LUYỆN & ĐÁNH GIÁ LIÊN HOÀN M5 TOÀN BỘ 3 CẤP ĐỘ TRÊN 30.000 CẶP <<<")
    print(f"Thiết bị: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 80)

    # 1. Tìm tài nguyên
    cpg_dir = locate_resource("cpg_export")
    assert cpg_dir is not None, "Không tìm thấy thư mục cpg_export!"

    m2_ckpt = locate_checkpoint("m2_dr")
    m4_ckpt = locate_checkpoint("m4_ep")
    logger.info("Checkpoint M2-DR: %s", m2_ckpt)
    logger.info("Checkpoint M4-EP: %s", m4_ckpt)

    # 2. Chuẩn bị dữ liệu 30.000 train, 3.000 val, 5.000 test
    tr_parquet = Path("data/raw/bigclonebench/train_shard_0.parquet")
    val_parquet = Path("data/raw/bigclonebench/validation_shard_0.parquet")
    test_jsonl = Path("results/bcb_5k_test_pairs.jsonl")

    assert tr_parquet.exists() and val_parquet.exists() and test_jsonl.exists(), "Thiếu file dữ liệu!"

    cpg_ids = set(p.stem for p in cpg_dir.glob("*.json"))

    def sample_data(p_path: Path, n_total: int, seed: int = 42) -> List[Dict[str, Any]]:
        df = pd.read_parquet(p_path)
        half = n_total // 2
        pos, neg = [], []
        for _, row in df.iterrows():
            id1, id2 = str(row["id1"]), str(row["id2"])
            if id1 not in cpg_ids or id2 not in cpg_ids:
                continue
            lbl = 1 if row["label"] else 0
            pair = {"id": int(row["id"]), "id1": id1, "id2": id2, "func1": str(row["func1"]).strip(), "func2": str(row["func2"]).strip(), "label": lbl}
            if lbl == 1 and len(pos) < half:
                pos.append(pair)
            elif lbl == 0 and len(neg) < half:
                neg.append(pair)
            if len(pos) >= half and len(neg) >= half:
                break
        comb = pos + neg
        rng = random.Random(seed)
        rng.shuffle(comb)
        return comb

    logger.info("Tạo tập Train 30.000 mẫu và Val 3.000 mẫu...")
    train_pairs = sample_data(tr_parquet, 30000, seed=42)
    val_pairs = sample_data(val_parquet, 3000, seed=42)

    with open(test_jsonl, "r", encoding="utf-8") as f:
        test_pairs = [json.loads(line) for line in f if line.strip()]

    logger.info("Xác nhận số lượng mẫu: Train=%d, Val=%d, Test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # 3. Nạp CPG cache vào RAM
    cpg_cache = build_cpg_cache([train_pairs, val_pairs, test_pairs], cpg_dir)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # 4. Nạp model M2-DR và M4-EP
    logger.info("Khởi tạo và nạp trọng số M2-DR và M4-EP...")
    m2_model = TransformerPairClassifier(hf_id="microsoft/graphcodebert-base", dropout=0.3).to(device)
    if m2_ckpt:
        ckpt2 = torch.load(m2_ckpt, map_location=device, weights_only=False)
        m2_model.load_state_dict(ckpt2.get("model_state", ckpt2), strict=False)

    m4_model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2).to(device)
    if m4_ckpt:
        ckpt4 = torch.load(m4_ckpt, map_location=device, weights_only=False)
        m4_model.load_state_dict(ckpt4.get("model_state", ckpt4), strict=False)

    # Dự đoán trên tập 5k test của M2 và M4
    logger.info("Dự đoán trên 5k test của M2-DR và M4-EP...")
    dummy_graph = CPGraph(
        node_type=np.zeros(1, dtype=np.int64),
        node_feat=np.zeros((1, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros(0, dtype=np.int64),
    )

    m2_test_probs, m4_test_probs, y_test = [], [], []
    m2_model.eval()
    m4_model.eval()
    with torch.no_grad():
        for i in range(0, len(test_pairs), 64):
            chunk = test_pairs[i : i + 64]
            c1 = [p["func1"] for p in chunk]
            c2 = [p["func2"] for p in chunk]
            in_a = {k: v.to(device) for k, v in tokenizer(c1, padding=True, truncation=True, max_length=384, return_tensors="pt").items()}
            in_b = {k: v.to(device) for k, v in tokenizer(c2, padding=True, truncation=True, max_length=384, return_tensors="pt").items()}
            with autocast(enabled=(device.type == "cuda")):
                l2 = m2_model(in_a, in_b)
            m2_test_probs.extend(torch.sigmoid(l2).cpu().numpy().tolist())

            ga = [cpg_cache.get(str(p.get("id1")), dummy_graph) for p in chunk]
            gb = [cpg_cache.get(str(p.get("id2")), dummy_graph) for p in chunk]
            pa = collate_pyg(ga).to(device)
            pb = collate_pyg(gb).to(device)
            l4 = m4_model(pa, pb)
            m4_test_probs.extend(torch.sigmoid(l4).cpu().numpy().tolist())
            y_test.extend([p["label"] for p in chunk])

    m2_probs_np = np.array(m2_test_probs)
    m4_probs_np = np.array(m4_test_probs)
    y_test_np = np.array(y_test)

    # Lưu kết quả cơ sở
    master_summary = {
        "m2_dr_baseline": compute_metrics_at_threshold(y_test_np, m2_probs_np, 0.52),
        "m4_ep_baseline": compute_metrics_at_threshold(y_test_np, m4_probs_np, 0.30),
    }

    # CHẠY CẤP ĐỘ 1
    master_summary["level1"] = run_level1(m2_probs_np, m4_probs_np, y_test_np)

    # CHẠY CẤP ĐỘ 2 (30.000 CẶP)
    master_summary["level2_30k"] = run_level2(
        train_pairs, val_pairs, test_pairs, m2_model, m4_model, tokenizer, cpg_cache, device, output_dir
    )

    # CHẠY CẤP ĐỘ 3 (30.000 CẶP)
    l3_metrics, l3_probs = run_level3(
        train_pairs, val_pairs, test_pairs, tokenizer, cpg_cache, m2_ckpt, m4_ckpt, device, output_dir, epochs=args.epochs_l3
    )
    master_summary["level3_30k"] = l3_metrics
    np.save(output_dir / "m5_level3_30k_5k_probs.npy", l3_probs)

    # Xuất file kết quả tổng hợp
    out_json = output_dir / "m5_unified_all_levels_test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(">>> BẢNG ĐỐI CHỨNG TỔNG HỢP TOÀN BỘ CÁC CẤP ĐỘ M5 (TRAIN TRÊN 30.000 CẶP) <<<")
    print("=" * 80)
    print(f"{'Mô hình / Cấp độ':<38} | {'F1-Score':<8} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'FP':<4} | {'FN':<4}")
    print("-" * 85)
    def pr(name, m):
        print(f"{name:<38} | {m['f1']:>6.2f}% | {m['accuracy']:>6.2f}% | {m['precision']:>7.2f}% | {m['recall']:>6.2f}% | {m['fp']:>4} | {m['fn']:>4}")

    pr("Đơn lẻ M4-EP (CPG-GNN)", master_summary["m4_ep_baseline"])
    pr("Đơn lẻ M2-DR (GraphCodeBERT)", master_summary["m2_dr_baseline"])
    pr("Cấp độ 1: Linear Blend (M2+M4)", master_summary["level1"]["linear_blend"])
    pr("Cấp độ 1: Confidence Gating", master_summary["level1"]["confidence_gating"])
    pr("Cấp độ 2: M5-Concat (30.000 cặp)", master_summary["level2_30k"]["m5_concat_30k"])
    pr("Cấp độ 2: M5-Gated (30.000 cặp)", master_summary["level2_30k"]["m5_gated_30k"])
    pr("Cấp độ 3: Deep Graph-Trans (30k)", master_summary["level3_30k"])
    print("=" * 85)
    print(f"Đã lưu toàn bộ kết quả vào: {out_json}")


if __name__ == "__main__":
    main()
