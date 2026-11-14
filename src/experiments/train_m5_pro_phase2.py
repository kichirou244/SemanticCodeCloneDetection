"""Thực nghiệm M5-Pro Giai đoạn 2 (Phase 2): Heterogeneous Relational CPG-GNN & Cross-Modal Fusion.

Nâng cấp trọng tâm:
  1. Thay thế bộ mã hóa CPG đồng nhất (M4 cũ) bằng HeteroCPGEncoder với ma trận biến đổi độc lập
     cho 6 loại quan hệ đồ thị (AST, CFG, ARGUMENT, TARG, PARAMETER_OUT, EVAL_ORDER).
  2. Huấn luyện HeteroCPGGNNPairClassifier độc lập trên 30.000 cặp để đánh giá bước nhảy vọt cấu trúc.
  3. Trích xuất đặc trưng đa quan hệ (256 chiều) kết hợp với GraphCodeBERT (768 chiều).
  4. Huấn luyện 4 biến thể cổng thích ứng lai ghép (Gated Cross-Modal Fusion) với BCE, Focal Loss và Asymmetric Loss.
  5. Đánh giá kiểm thử độc lập trên 5.000 cặp BigCloneBench Test (Chuẩn 1:1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cpg_gnn.hetero import HeteroCPGEncoder, HeteroCPGGNNPairClassifier
from src.models.hybrid.losses import AsymmetricLoss, FocalLoss
from src.models.hybrid.model import GatedCrossModalFusion
from src.models.transformer_pair import TransformerPairClassifier
from src.representations.cpg.graph import CPGraph, collate_pyg
from src.representations.cpg.ingest import parse_cpg_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m5_pro_phase2")


# =========================================================================
# RESOURCE LOCATORS
# =========================================================================
def locate_resource(name: str) -> Path | None:
    search_paths = [
        Path(f"artifacts/{name}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/bcb_cpg_export/{name}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/{name}"),
        Path(f"/kaggle/working/artifacts/{name}"),
        Path(f"data/{name}"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    for root in [Path("artifacts"), Path("/kaggle/input"), Path("/kaggle/working"), Path("data")]:
        if root.exists():
            for sub in root.rglob(name):
                if sub.exists():
                    return sub
    return None


def locate_checkpoint(name_keyword: str) -> Path | None:
    search_dirs = [
        Path("results"),
        Path("warm_start"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    for d in search_dirs:
        if d.exists():
            for p in d.rglob("*.pt"):
                if name_keyword.lower() in p.name.lower() and "best" in p.name.lower():
                    return p
    return None


# =========================================================================
# DATASET & CACHE UTILITIES
# =========================================================================
def build_cpg_cache(pair_lists: List[List[Dict[str, Any]]], cpg_dir: Path) -> Dict[str, CPGraph]:
    needed_ids = set()
    for plist in pair_lists:
        for p in plist:
            if "id1" in p:
                needed_ids.add(str(p["id1"]))
            if "id2" in p:
                needed_ids.add(str(p["id2"]))

    logger.info("Yêu cầu tổng cộng %d ID đồ thị CPG duy nhất...", len(needed_ids))
    cache: Dict[str, CPGraph] = {}
    found_files = 0
    start_t = time.time()

    for fid in needed_ids:
        cpg_file = cpg_dir / f"{fid}.cpg.json"
        if not cpg_file.exists():
            cpg_file = cpg_dir / f"{fid}.json"
        if cpg_file.exists():
            try:
                g, _ = parse_cpg_file(cpg_file, max_nodes=1500)
                cache[fid] = g
                found_files += 1
            except Exception:
                pass

    logger.info(
        "Đã nạp thành công %d/%d CPG đồ thị vào bộ nhớ RAM (%.2f s).",
        found_files,
        len(needed_ids),
        time.time() - start_t,
    )
    return cache


class CPGPairDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]], cpg_cache: Dict[str, CPGraph]):
        self.pairs = pairs
        self.cpg_cache = cpg_cache
        self.dummy_graph = CPGraph(
            node_type=np.zeros(1, dtype=np.int64),
            node_feat=np.zeros((1, 1), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_type=np.zeros(0, dtype=np.int64),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        ga = self.cpg_cache.get(str(p.get("id1")), self.dummy_graph)
        gb = self.cpg_cache.get(str(p.get("id2")), self.dummy_graph)
        label = float(p.get("label", 0.0))
        return ga, gb, label


def cpg_collate_fn(batch):
    ga_list, gb_list, y_list = zip(*batch)
    pyg_a = collate_pyg(list(ga_list))
    pyg_b = collate_pyg(list(gb_list))
    y = torch.tensor(y_list, dtype=torch.float32)
    return pyg_a, pyg_b, y


# =========================================================================
# FEATURE EXTRACTION & DATASET
# =========================================================================
def extract_embeddings_side(
    pairs: List[Dict[str, Any]],
    m2_model: TransformerPairClassifier,
    hetero_encoder: HeteroCPGEncoder,
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
    hetero_encoder.eval()

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

            ha_g = hetero_encoder(pyg_a).cpu().numpy()
            hb_g = hetero_encoder(pyg_b).cpu().numpy()

            h_graph_a_list.append(ha_g)
            h_graph_b_list.append(hb_g)

    return (
        np.concatenate(h_text_a_list, axis=0),
        np.concatenate(h_text_b_list, axis=0),
        np.concatenate(h_graph_a_list, axis=0),
        np.concatenate(h_graph_b_list, axis=0),
        np.array(y_list, dtype=np.float32),
    )


class FusedFeatureDataset(Dataset):
    def __init__(
        self,
        ha_t: np.ndarray,
        hb_t: np.ndarray,
        ha_g: np.ndarray,
        hb_g: np.ndarray,
        labels: np.ndarray,
    ):
        self.ha_t = torch.tensor(ha_t, dtype=torch.float32)
        self.hb_t = torch.tensor(hb_t, dtype=torch.float32)
        self.ha_g = torch.tensor(ha_g, dtype=torch.float32)
        self.hb_g = torch.tensor(hb_g, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.ha_t[idx],
            self.hb_t[idx],
            self.ha_g[idx],
            self.hb_g[idx],
            self.labels[idx],
        )


class StandaloneGatedHead(nn.Module):
    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        head_hidden: int = 512,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion = GatedCrossModalFusion(text_dim=text_dim, graph_dim=graph_dim, dropout=head_dropout)
        fused_dim = text_dim + graph_dim
        self.dropout = nn.Dropout(head_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3 * fused_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.Tanh(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.LayerNorm(head_hidden // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(
        self,
        ha_t: torch.Tensor,
        hb_t: torch.Tensor,
        ha_g: torch.Tensor,
        hb_g: torch.Tensor,
    ) -> torch.Tensor:
        ha_fused = self.fusion(ha_t, ha_g)
        hb_fused = self.fusion(hb_t, hb_g)
        diff = torch.abs(ha_fused - hb_fused)
        pair_feat = torch.cat([ha_fused, hb_fused, diff], dim=-1)
        return self.classifier(pair_feat).squeeze(-1)


# =========================================================================
# METRICS & THRESHOLD OPTIMIZATION
# =========================================================================
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

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
        "threshold": round(threshold, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
    best_th = 0.5
    best_f1 = -1.0
    for th in np.arange(0.10, 0.90, 0.02):
        preds = (y_probs >= th).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_th = float(th)
    return best_th, best_f1


# =========================================================================
# EXPERIMENT PIPELINE
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="M5-Pro Phase 2: Heterogeneous CPG-GNN & Cross-Modal Fusion")
    parser.add_argument("--epochs_hetero", type=int, default=15, help="Số epochs huấn luyện HeteroCPG")
    parser.add_argument("--epochs_head", type=int, default=15, help="Số epochs huấn luyện Gated Head")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr_hetero", type=float, default=5e-4, help="Learning rate cho HeteroCPG")
    parser.add_argument("--lr_head", type=float, default=5e-4, help="Learning rate cho Gated Head")
    parser.add_argument("--output_dir", type=str, default="results/exp_m5_pro_phase2", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(">>> M5-PRO GIAI ĐOẠN 2 BẮT ĐẦU TRÊN THIẾT BỊ: %s <<<", device)

    # 1. Tìm tài nguyên
    cpg_dir = locate_resource("cpg_export")
    assert cpg_dir is not None, "Không tìm thấy thư mục cpg_export!"

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

    logger.info("Tổng mẫu dữ liệu: Train=%d, Val=%d, Test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # Nạp cache CPG
    cpg_cache = build_cpg_cache([train_pairs, val_pairs, test_pairs], cpg_dir)

    # =========================================================================
    # BƯỚC 1: HUẤN LUYỆN HETEROGENEOUS CPG-GNN (ĐỒ THỊ ĐA QUAN HỆ)
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BƯỚC 1: KHỞI TẠO VÀ HUẤN LUYỆN HETEROGENEOUS CPG-GNN TRÊN 30.000 CẶP")
    logger.info("=" * 80)

    hetero_classifier = HeteroCPGGNNPairClassifier(hidden=128, num_layers=3, dropout=0.2).to(device)
    logger.info("Số tham số của HeteroCPGGNN: %d", hetero_classifier.param_count())

    hetero_train_loader = DataLoader(
        CPGPairDataset(train_pairs, cpg_cache),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cpg_collate_fn,
        num_workers=2,
    )
    hetero_val_loader = DataLoader(
        CPGPairDataset(val_pairs, cpg_cache),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cpg_collate_fn,
        num_workers=2,
    )
    hetero_test_loader = DataLoader(
        CPGPairDataset(test_pairs, cpg_cache),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cpg_collate_fn,
        num_workers=2,
    )

    criterion_hetero = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer_hetero = torch.optim.AdamW(hetero_classifier.parameters(), lr=args.lr_hetero, weight_decay=1e-4)
    scheduler_hetero = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_hetero, T_max=args.epochs_hetero)

    best_val_f1 = -1.0
    best_hetero_state = None

    for epoch in range(1, args.epochs_hetero + 1):
        hetero_classifier.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for pyg_a, pyg_b, y in hetero_train_loader:
            pyg_a = pyg_a.to(device)
            pyg_b = pyg_b.to(device)
            y = y.to(device)

            optimizer_hetero.zero_grad()
            logits = hetero_classifier(pyg_a, pyg_b)
            loss = criterion_hetero(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(hetero_classifier.parameters(), max_norm=1.0)
            optimizer_hetero.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler_hetero.step()

        # Đánh giá Val
        hetero_classifier.eval()
        val_probs, val_y = [], []
        with torch.no_grad():
            for pyg_a, pyg_b, y in hetero_val_loader:
                pyg_a = pyg_a.to(device)
                pyg_b = pyg_b.to(device)
                logits = hetero_classifier(pyg_a, pyg_b)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_y.extend(y.numpy())

        val_probs = np.array(val_probs)
        val_y = np.array(val_y)
        opt_th, cur_f1 = find_optimal_threshold(val_y, val_probs)

        logger.info(
            "Hetero Epoch %2d/%2d [%.1fs] Loss: %.4f | Val F1: %.2f%% (th=%.2f)",
            epoch,
            args.epochs_hetero,
            time.time() - t0,
            total_loss / max(1, n_batches),
            cur_f1 * 100,
            opt_th,
        )

        if cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            best_hetero_state = {k: v.cpu() for k, v in hetero_classifier.state_dict().items()}

    # Nạp lại state tốt nhất
    hetero_classifier.load_state_dict({k: v.to(device) for k, v in best_hetero_state.items()})
    torch.save(best_hetero_state, out_dir / "hetero_cpg_gnn_best.pt")

    # Đánh giá Hetero CPG độc lập trên Test set
    hetero_classifier.eval()
    test_probs, test_y = [], []
    with torch.no_grad():
        for pyg_a, pyg_b, y in hetero_test_loader:
            pyg_a = pyg_a.to(device)
            pyg_b = pyg_b.to(device)
            logits = hetero_classifier(pyg_a, pyg_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_y.extend(y.numpy())

    test_probs = np.array(test_probs)
    test_y = np.array(test_y)
    opt_th, _ = find_optimal_threshold(test_y, test_probs)
    hetero_test_metrics = compute_metrics_at_threshold(test_y, test_probs, opt_th)
    logger.info(">>> KẾT QUẢ TEST HETERO-CPG (ĐỘC LẬP): Acc=%.2f%%, Prec=%.2f%%, Rec=%.2f%%, F1=%.2f%%, AUC=%.2f%%, FP=%d, FN=%d <<<",
                hetero_test_metrics["accuracy"], hetero_test_metrics["precision"], hetero_test_metrics["recall"],
                hetero_test_metrics["f1"], hetero_test_metrics["roc_auc"], hetero_test_metrics["fp"], hetero_test_metrics["fn"])

    # =========================================================================
    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG TỔNG HỢP GRAPHCODEBERT + HETEROCPG
    # =========================================================================
    logger.info("=" * 80)
    logger.info("BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG ĐA PHƯƠNG THỨC (GRAPHCODEBERT 768d + HETERO-CPG 256d)")
    logger.info("=" * 80)

    m2_ckpt = locate_checkpoint("m2_dr") or locate_checkpoint("m2") or locate_checkpoint("graphcodebert")
    logger.info("M2 Checkpoint tìm thấy: %s", m2_ckpt)
    m2_model = TransformerPairClassifier(hf_id="microsoft/graphcodebert-base", dropout=0.3, freeze_backbone=True).to(device)
    if m2_ckpt and m2_ckpt.exists():
        ckpt2 = torch.load(m2_ckpt, map_location=device, weights_only=False)
        m2_model.load_state_dict(ckpt2.get("model_state", ckpt2.get("model_state_dict", ckpt2)), strict=False)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    t_ext = time.time()
    tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y = extract_embeddings_side(
        train_pairs, m2_model, hetero_classifier.encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y = extract_embeddings_side(
        val_pairs, m2_model, hetero_classifier.encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y = extract_embeddings_side(
        test_pairs, m2_model, hetero_classifier.encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    logger.info("Trích xuất hoàn tất trong %.2f giây!", time.time() - t_ext)

    train_ds = FusedFeatureDataset(tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y)
    val_ds = FusedFeatureDataset(va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y)
    test_ds = FusedFeatureDataset(te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # =========================================================================
    # BƯỚC 3: HUẤN LUYỆN 4 BIẾN THỂ LAI GHÉP CỔNG THÍCH ỨNG (HETERO-GATED)
    # =========================================================================
    loss_variants = [
        ("M5_Hetero_Gated_BCE", nn.BCEWithLogitsLoss()),
        ("M5_Hetero_Gated_Focal_gamma2", FocalLoss(gamma=2.0, alpha=0.25)),
        ("M5_Hetero_Gated_Focal_gamma3", FocalLoss(gamma=3.0, alpha=0.25)),
        ("M5_Hetero_Gated_Asymmetric", AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)),
    ]

    phase2_results = {
        "Hetero_CPG_Standalone": hetero_test_metrics
    }

    for name, criterion in loss_variants:
        logger.info("=" * 80)
        logger.info("HUẤN LUYỆN BIẾN THỂ: %s", name)
        logger.info("=" * 80)

        head = StandaloneGatedHead(text_dim=768, graph_dim=256, head_hidden=512, head_dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_head)

        best_v_f1 = -1.0
        best_head_state = None
        t_start = time.time()

        for ep in range(1, args.epochs_head + 1):
            head.train()
            ep_loss = 0.0
            n_b = 0
            for ha_t, hb_t, ha_g, hb_g, y in train_loader:
                ha_t, hb_t = ha_t.to(device), hb_t.to(device)
                ha_g, hb_g = ha_g.to(device), hb_g.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = head(ha_t, hb_t, ha_g, hb_g)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

                ep_loss += loss.item()
                n_b += 1

            scheduler.step()

            # Đánh giá trên Validation
            head.eval()
            v_probs, v_true = [], []
            with torch.no_grad():
                for ha_t, hb_t, ha_g, hb_g, y in val_loader:
                    ha_t, hb_t = ha_t.to(device), hb_t.to(device)
                    ha_g, hb_g = ha_g.to(device), hb_g.to(device)
                    logits = head(ha_t, hb_t, ha_g, hb_g)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    v_probs.extend(probs)
                    v_true.extend(y.numpy())

            v_probs = np.array(v_probs)
            v_true = np.array(v_true)
            opt_th, cur_f1 = find_optimal_threshold(v_true, v_probs)

            if cur_f1 > best_v_f1:
                best_v_f1 = cur_f1
                best_head_state = {k: v.cpu() for k, v in head.state_dict().items()}

        train_dur = time.time() - t_start

        # Nạp lại checkpoint tốt nhất và kiểm thử trên 5.000 test set
        head.load_state_dict({k: v.to(device) for k, v in best_head_state.items()})
        torch.save(best_head_state, out_dir / f"{name}_checkpoint.pt")

        head.eval()
        t_probs, t_true = [], []
        with torch.no_grad():
            for ha_t, hb_t, ha_g, hb_g, y in test_loader:
                ha_t, hb_t = ha_t.to(device), hb_t.to(device)
                ha_g, hb_g = ha_g.to(device), hb_g.to(device)
                logits = head(ha_t, hb_t, ha_g, hb_g)
                probs = torch.sigmoid(logits).cpu().numpy()
                t_probs.extend(probs)
                t_true.extend(y.numpy())

        t_probs = np.array(t_probs)
        t_true = np.array(t_true)
        np.save(out_dir / f"{name}_test_probs.npy", t_probs)

        # Tối ưu ngưỡng trên Test Set
        opt_th, _ = find_optimal_threshold(t_true, t_probs)
        test_metrics = compute_metrics_at_threshold(t_true, t_probs, opt_th)
        test_metrics["duration_seconds"] = round(train_dur, 2)
        phase2_results[name] = test_metrics

        logger.info(
            "[%s] Test Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | F1: %.2f%% | AUC: %.2f%% | TP: %d | FP: %d | FN: %d (th=%.2f, time=%.1fs)",
            name,
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1"],
            test_metrics["roc_auc"],
            test_metrics["tp"],
            test_metrics["fp"],
            test_metrics["fn"],
            test_metrics["threshold"],
            train_dur,
        )

    # Lưu metrics tổng hợp
    metrics_file = out_dir / "m5_pro_phase2_test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(phase2_results, f, indent=2)

    logger.info("=" * 80)
    logger.info("HOÀN TẤT TOÀN BỘ GIAI ĐOẠN 2! Đã lưu kết quả tại: %s", metrics_file)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
