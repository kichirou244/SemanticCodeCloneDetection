"""Thực nghiệm M5-Pro Giai đoạn 1 (Phase 1): Tối ưu hóa Hàm Mất mát & Khai phá Mẫu khó.

Mục tiêu cốt lõi:
  Triệt tiêu 130 ca False Positives của M5-Gated (30k) mà không làm suy giảm kỷ lục Recall 96.72%.
  Huấn luyện và so sánh 4 biến thể mất mát trên đúng 30.000 cặp dữ liệu BigCloneBench:
    1. M5_Gated_BCE: Hàm mất mát chuẩn nhị phân (Baseline hiện tại).
    2. M5_Gated_Focal_gamma2: Focal Loss (gamma=2.0, alpha=0.25).
    3. M5_Gated_Focal_gamma3: Focal Loss (gamma=3.0, alpha=0.25) phạt mạnh hơn vào hard negatives.
    4. M5_Gated_Asymmetric_Loss: Asymmetric Loss (gamma_neg=4.0, gamma_pos=1.0) phạt lũy thừa ca Non-clone đoán sai.
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
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cpg_gnn.model import CPGEncoder, CPGGNNPairClassifier
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
logger = logging.getLogger("m5_pro_phase1")


# =========================================================================
# RESOURCE LOCATORS
# =========================================================================
def locate_resource(name: str) -> Path | None:
    search_paths = [
        Path(f"artifacts/{name}"),
        Path("artifacts/cpg_export"),
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
        np.array(y_list, dtype=np.float32),
    )


# =========================================================================
# FEATURE DATASET & GATED HEAD ARCHITECTURE
# =========================================================================
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
# TRAINING ENGINE
# =========================================================================
def train_and_eval_head(
    model_name: str,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_val: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
) -> Tuple[Dict[str, Any], np.ndarray, nn.Module]:
    logger.info("--- Huấn luyện %s (Epochs=%d, LR=%s) ---", model_name, epochs, lr)
    head = StandaloneGatedHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1 = -1.0
    best_state = None
    best_th = 0.5

    for epoch in range(1, epochs + 1):
        head.train()
        train_loss = 0.0
        for ha_t, hb_t, ha_g, hb_g, y in train_loader:
            ha_t = ha_t.to(device)
            hb_t = hb_t.to(device)
            ha_g = ha_g.to(device)
            hb_g = hb_g.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = head(ha_t, hb_t, ha_g, hb_g)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_loader.dataset)

        # Validation
        head.eval()
        val_probs = []
        with torch.no_grad():
            for ha_t, hb_t, ha_g, hb_g, _ in val_loader:
                ha_t = ha_t.to(device)
                hb_t = hb_t.to(device)
                ha_g = ha_g.to(device)
                hb_g = hb_g.to(device)
                logits = head(ha_t, hb_t, ha_g, hb_g)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)

        val_probs = np.concatenate(val_probs, axis=0)
        th, val_f1 = find_optimal_threshold(y_val, val_probs)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_th = th
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        logger.info(
            "[%s] Epoch %2d/%2d | Train Loss: %.4f | Val F1: %.2f%% (Optimal th=%.2f)",
            model_name,
            epoch,
            epochs,
            train_loss,
            val_f1 * 100,
            th,
        )

    # Đánh giá trên tập kiểm thử độc lập 5.000 cặp
    head.load_state_dict(best_state)
    head.eval()
    test_probs = []
    with torch.no_grad():
        for ha_t, hb_t, ha_g, hb_g, _ in test_loader:
            ha_t = ha_t.to(device)
            hb_t = hb_t.to(device)
            ha_g = ha_g.to(device)
            hb_g = hb_g.to(device)
            logits = head(ha_t, hb_t, ha_g, hb_g)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)

    test_probs = np.concatenate(test_probs, axis=0)
    test_metrics = compute_metrics_at_threshold(y_test, test_probs, best_th)
    logger.info(
        "==> [%s TEST] F1: %.2f%% | Acc: %.2f%% | Prec: %.2f%% | Rec: %.2f%% | AUC: %.2f%% | FP: %d | FN: %d",
        model_name,
        test_metrics["f1"],
        test_metrics["accuracy"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["roc_auc"],
        test_metrics["fp"],
        test_metrics["fn"],
    )
    return test_metrics, test_probs, head


# =========================================================================
# MAIN PIPELINE
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="M5-Pro Phase 1: Tối ưu hóa Hàm Mất mát & Khai phá Mẫu khó trên 30.000 cặp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/exp_m5_pro_phase1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print(">>> THỰC NGHIỆM M5-PRO PHASE 1: TỐI ƯU HÀM MẤT MÁT TRÊN 30.000 CẶP <<<")
    print(f"Thiết bị: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 80)

    # 1. Tìm tài nguyên
    cpg_dir = locate_resource("cpg_export")
    assert cpg_dir is not None, "Không tìm thấy thư mục cpg_export!"

    m2_ckpt = locate_checkpoint("m2_dr")
    m4_ckpt = locate_checkpoint("m4_ep")
    logger.info("CPG Dir: %s", cpg_dir)
    logger.info("Checkpoint M2-DR: %s", m2_ckpt)
    logger.info("Checkpoint M4-EP: %s", m4_ckpt)

    # 2. Nạp dữ liệu 30.000 train, 3.000 val, 5.000 test
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

    # 5. Trích xuất vector nhúng
    t_ext_0 = time.perf_counter()
    logger.info(">>> Trích xuất đặc trưng cho Train (%d cặp)...", len(train_pairs))
    tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y = extract_embeddings_side(
        train_pairs, m2_model, m4_model, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )

    logger.info(">>> Trích xuất đặc trưng cho Val (%d cặp)...", len(val_pairs))
    va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y = extract_embeddings_side(
        val_pairs, m2_model, m4_model, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )

    logger.info(">>> Trích xuất đặc trưng cho Test (%d cặp)...", len(test_pairs))
    te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y = extract_embeddings_side(
        test_pairs, m2_model, m4_model, tokenizer, cpg_cache, device, batch_size=args.batch_size
    )
    t_ext_1 = time.perf_counter()
    logger.info("Đã hoàn tất trích xuất đặc trưng toàn bộ 3 phân tập trong %.2f phút!", (t_ext_1 - t_ext_0) / 60)

    # Giải phóng GPU VRAM của M2 và M4
    del m2_model, m4_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_ds = FusedFeatureDataset(tr_ha_t, tr_hb_t, tr_ha_g, tr_hb_g, tr_y)
    val_ds = FusedFeatureDataset(va_ha_t, va_hb_t, va_ha_g, va_hb_g, va_y)
    test_ds = FusedFeatureDataset(te_ha_t, te_hb_t, te_ha_g, te_hb_g, te_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 6. Huấn luyện đối chứng 4 biến thể hàm mất mát
    experiments = [
        ("M5_Gated_BCE", nn.BCEWithLogitsLoss()),
        ("M5_Gated_Focal_gamma2", FocalLoss(alpha=0.25, gamma=2.0)),
        ("M5_Gated_Focal_gamma3", FocalLoss(alpha=0.25, gamma=3.0)),
        ("M5_Gated_Asymmetric_Loss", AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)),
    ]

    all_results = {}
    for name, criterion in experiments:
        t0 = time.perf_counter()
        metrics, probs, model = train_and_eval_head(
            name, criterion, train_loader, val_loader, test_loader, va_y, te_y, device, epochs=args.epochs
        )
        t1 = time.perf_counter()
        metrics["duration_seconds"] = round(t1 - t0, 2)
        all_results[name] = metrics

        # Lưu checkpoint & probabilities
        torch.save(model.state_dict(), output_dir / f"{name}_checkpoint.pt")
        np.save(output_dir / f"{name}_test_probs.npy", probs)

    # 7. Ghi kết quả tổng hợp
    results_path = output_dir / "m5_pro_phase1_test_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Đã lưu kết quả thực nghiệm hoàn tất tại: %s", results_path)

    # In bảng so sánh trực tiếp
    print("\n" + "=" * 85)
    print("BẢNG TỔNG HỢP KẾT QUẢ M5-PRO PHASE 1 TRÊN 5.000 TEST BIGCLONEBENCH (30K TRAIN)")
    print("=" * 85)
    print(f"{'Biến thể Mất mát':<28} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9} | {'FP':<5} | {'FN':<5}")
    print("-" * 85)
    for name, m in all_results.items():
        print(
            f"{name:<28} | {m['accuracy']:>7.2f}% | {m['precision']:>7.2f}% | "
            f"{m['recall']:>7.2f}% | {m['f1']:>7.2f}% | {m['roc_auc']:>7.2f}% | "
            f"{m['fp']:>5} | {m['fn']:>5}"
        )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
