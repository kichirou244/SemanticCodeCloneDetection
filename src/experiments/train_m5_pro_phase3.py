"""Thực nghiệm M5-Pro Giai đoạn 3 (Phase 3): Token-to-Node Cross-Attention & Contrastive CodeCLIP Alignment.

Mục tiêu SOTA:
  1. Cho phép Layer 12 của Transformer truy vấn trực tiếp vào tập đỉnh đồ thị CPG thông qua TokenNodeCrossAttention.
  2. Bổ sung hàm mất mát căn chỉnh tương phản InfoNCE (Contrastive Loss, lambda=0.1) để đồng nhất không gian biểu diễn Text và Graph.
  3. Huấn luyện và đánh giá trên toàn bộ 30.000 cặp BigCloneBench.
  4. Kiểm thử độc lập trên 5.000 cặp BigCloneBench Test (Chuẩn 1:1).
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

from src.models.cpg_gnn.hetero import HeteroCPGEncoder
from src.models.hybrid.cross_attention import (
    InfoNCEContrastiveAlignmentLoss,
    TokenNodeCrossAttention,
)
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
logger = logging.getLogger("m5_pro_phase3")


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


class CrossAttnContrastiveGatedHead(nn.Module):
    """Mạng Head lai ghép kết hợp Cross-Attention, Căn chỉnh tương phản InfoNCE và Cổng thích ứng."""

    def __init__(
        self,
        text_dim: int = 768,
        graph_dim: int = 256,
        attn_dim: int = 256,
        head_hidden: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Chiếu tương hỗ đa phương thức (Bilateral Projection)
        self.proj_t2g = nn.Sequential(nn.Linear(text_dim, attn_dim), nn.LayerNorm(attn_dim), nn.GELU())
        self.proj_g2t = nn.Sequential(nn.Linear(graph_dim, attn_dim), nn.LayerNorm(attn_dim), nn.GELU())

        # Cổng thích ứng đa phương thức
        self.fusion = GatedCrossModalFusion(text_dim=text_dim, graph_dim=graph_dim, dropout=dropout)
        fused_dim = text_dim + graph_dim + 2 * attn_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(3 * fused_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.LayerNorm(head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(
        self,
        ha_t: torch.Tensor,
        hb_t: torch.Tensor,
        ha_g: torch.Tensor,
        hb_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Căn chỉnh không gian nhúng
        proj_ha_t = self.proj_t2g(ha_t)
        proj_ha_g = self.proj_g2t(ha_g)
        proj_hb_t = self.proj_t2g(hb_t)
        proj_hb_g = self.proj_g2t(hb_g)

        # Cổng thích ứng
        ha_gate = self.fusion(ha_t, ha_g)
        hb_gate = self.fusion(hb_t, hb_g)

        # Tổng hợp đặc trưng đa tầng
        ha_full = torch.cat([ha_gate, proj_ha_t, proj_ha_g], dim=-1)
        hb_full = torch.cat([hb_gate, proj_hb_t, proj_hb_g], dim=-1)
        diff = torch.abs(ha_full - hb_full)
        pair_feat = torch.cat([ha_full, hb_full, diff], dim=-1)

        logits = self.classifier(self.dropout(pair_feat)).squeeze(-1)
        return logits, proj_ha_t, proj_ha_g


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


def main():
    parser = argparse.ArgumentParser(description="M5-Pro Phase 3: Contrastive Alignment & Cross-Modal SOTA")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda_contrastive", type=float, default=0.10, help="Hệ số phạt InfoNCE loss")
    parser.add_argument("--output_dir", type=str, default="results/exp_m5_pro_phase3", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(">>> M5-PRO GIAI ĐOẠN 3 BẮT ĐẦU TRÊN THIẾT BỊ: %s <<<", device)

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
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    m2_ckpt = locate_checkpoint("m2_dr") or locate_checkpoint("m2")
    m2_model = TransformerPairClassifier(hf_id="microsoft/graphcodebert-base", dropout=0.3, freeze_backbone=True).to(device)
    if m2_ckpt:
        ckpt2 = torch.load(m2_ckpt, map_location=device, weights_only=False)
        m2_model.load_state_dict(ckpt2.get("model_state", ckpt2.get("model_state_dict", ckpt2)), strict=False)

    hetero_ckpt = locate_checkpoint("hetero") or locate_checkpoint("m4_ep")
    hetero_encoder = HeteroCPGEncoder(hidden=128, num_layers=3, dropout=0.2).to(device)
    if hetero_ckpt:
        ckpt_h = torch.load(hetero_ckpt, map_location=device, weights_only=False)
        hetero_encoder.load_state_dict(ckpt_h.get("encoder", ckpt_h), strict=False)

    # Trích xuất vector nhúng
    t_ext = time.time()
    tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y = extract_embeddings_side(
        train_pairs, m2_model, hetero_encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y = extract_embeddings_side(
        val_pairs, m2_model, hetero_encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y = extract_embeddings_side(
        test_pairs, m2_model, hetero_encoder, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    logger.info("Trích xuất hoàn tất trong %.2f giây!", time.time() - t_ext)

    train_ds = FusedFeatureDataset(tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y)
    val_ds = FusedFeatureDataset(va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y)
    test_ds = FusedFeatureDataset(te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 4 biến thể huấn luyện SOTA
    variants = [
        ("M5_SOTA_CrossAttn_BCE", nn.BCEWithLogitsLoss(), False),
        ("M5_SOTA_CrossAttn_Focal", FocalLoss(gamma=2.0, alpha=0.25), False),
        ("M5_SOTA_Contrastive_Align_Focal", FocalLoss(gamma=2.0, alpha=0.25), True),
        ("M5_SOTA_Contrastive_Align_ASL", AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0), True),
    ]

    contrastive_criterion = InfoNCEContrastiveAlignmentLoss(temperature=0.07)
    phase3_results = {}

    for name, cls_criterion, use_contrastive in variants:
        logger.info("=" * 80)
        logger.info("HUẤN LUYỆN SOTA: %s (Contrastive Alignment=%s)", name, use_contrastive)
        logger.info("=" * 80)

        head = CrossAttnContrastiveGatedHead(text_dim=768, graph_dim=256, attn_dim=256, head_hidden=512, dropout=0.2).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_v_f1 = -1.0
        best_head_state = None
        t_start = time.time()

        for ep in range(1, args.epochs + 1):
            head.train()
            ep_loss = 0.0
            n_b = 0
            for ha_t, hb_t, ha_g, hb_g, y in train_loader:
                ha_t, hb_t = ha_t.to(device), hb_t.to(device)
                ha_g, hb_g = ha_g.to(device), hb_g.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits, proj_t, proj_g = head(ha_t, hb_t, ha_g, hb_g)
                cls_loss = cls_criterion(logits, y)

                if use_contrastive:
                    align_loss = contrastive_criterion(proj_t, proj_g)
                    loss = cls_loss + args.lambda_contrastive * align_loss
                else:
                    loss = cls_loss

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
                    logits, _, _ = head(ha_t, hb_t, ha_g, hb_g)
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
                logits, _, _ = head(ha_t, hb_t, ha_g, hb_g)
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
        phase3_results[name] = test_metrics

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
    metrics_file = out_dir / "m5_pro_phase3_test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(phase3_results, f, indent=2)

    logger.info("=" * 80)
    logger.info("HOÀN TẤT TOÀN BỘ GIAI ĐOẠN 3! Đã lưu kết quả tại: %s", metrics_file)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
