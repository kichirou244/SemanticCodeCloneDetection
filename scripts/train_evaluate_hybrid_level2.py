#!/usr/bin/env python3
"""Huấn luyện và Đánh giá Mô hình Lai Đặc trưng M5 Cấp độ 2 (Feature-Level Soft Fusion).

So sánh 2 kiến trúc lai ghép đặc trưng:
  1. Concat-MLP Fusion (Baseline M5)
  2. Gated Cross-Modal Soft Fusion (Proposed M5-Gated)

Tập dữ liệu:
  - Train: results/bcb_hybrid_train_6k.jsonl (6.000 cặp)
  - Validation: results/bcb_hybrid_val_2k.jsonl (2.000 cặp)
  - Test: results/bcb_5k_test_pairs.jsonl (5.000 cặp chuẩn 1:1)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from src.models.hybrid.model import HybridPairClassifier
from src.data.pairs.bcb_dataset import collate_cpg_pairs, ClonePairSample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

M2_CKPT = ROOT / "results" / "exp_m2_dr" / "best_checkpoint.pt"
M4_CKPT = ROOT / "results" / "exp_m4_ep" / "best_checkpoint.pt"
CPG_EXPORT_DIR = ROOT / "artifacts" / "cpg_export"

TRAIN_FILE = ROOT / "results" / "bcb_hybrid_train_6k.jsonl"
VAL_FILE = ROOT / "results" / "bcb_hybrid_val_2k.jsonl"
TEST_FILE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"

CACHE_TRAIN = ROOT / "results" / "hybrid_cache_train_6k.npz"
CACHE_VAL = ROOT / "results" / "hybrid_cache_val_2k.npz"
CACHE_TEST = ROOT / "results" / "hybrid_cache_test_5k.npz"

RESULTS_JSON = ROOT / "results" / "hybrid_level2_results.json"


def load_pairs(filepath: Path) -> list[dict]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def extract_and_cache_embeddings(
    pairs: list[dict],
    cache_path: Path,
    m2_model: TransformerPairClassifier,
    m4_model: CPGGNNPairClassifier,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path.name}...")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    print(f"\n>>> Extracting embeddings for {len(pairs)} pairs ({cache_path.name})...")
    c1 = [p["func1"] for p in pairs]
    c2 = [p["func2"] for p in pairs]
    labels = np.array([p["label"] for p in pairs], dtype=int)

    sample_objects = [
        ClonePairSample(
            pair_id=str(p["id"]),
            id1=str(p["id1"]),
            id2=str(p["id2"]),
            func1=p["func1"],
            func2=p["func2"],
            label=int(p["label"]),
            language="java",
            split="train",
        )
        for p in pairs
    ]

    h_text_a_list, h_text_b_list = [], []
    h_graph_a_list, h_graph_b_list = [], []

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            # 1. Text Encoder (GraphCodeBERT - 768 dims)
            b1 = tokenizer(c1[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(c2[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            ha_text = m2_model.encode_side(b1).cpu().numpy()
            hb_text = m2_model.encode_side(b2).cpu().numpy()
            h_text_a_list.append(ha_text)
            h_text_b_list.append(hb_text)

            # 2. Graph Encoder (CPG-GNN - 256 dims)
            batch_samples = sample_objects[i : i + batch_size]
            pyg_a, pyg_b, _ = collate_cpg_pairs(batch_samples, cpg_store=CPG_EXPORT_DIR, max_nodes=1500, device=DEVICE)
            ha_graph = m4_model.encode_side(pyg_a).cpu().numpy()
            hb_graph = m4_model.encode_side(pyg_b).cpu().numpy()
            h_graph_a_list.append(ha_graph)
            h_graph_b_list.append(hb_graph)

            if (i // batch_size) % 25 == 0:
                print(f"  Processed {min(i + batch_size, len(pairs))}/{len(pairs)} pairs...", flush=True)

    emb_dict = {
        "text_a": np.concatenate(h_text_a_list, axis=0).astype(np.float32),
        "text_b": np.concatenate(h_text_b_list, axis=0).astype(np.float32),
        "graph_a": np.concatenate(h_graph_a_list, axis=0).astype(np.float32),
        "graph_b": np.concatenate(h_graph_b_list, axis=0).astype(np.float32),
        "labels": labels,
    }
    np.savez_compressed(cache_path, **emb_dict)
    print(f"  Saved {cache_path.name} in {time.perf_counter() - t0:.1f}s!")
    return emb_dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    auc = 0.5
    if scores is not None:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = 0.5

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
    }


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    best_th = 0.5
    best_f1 = -1.0
    best_m = None
    for th in np.linspace(0.01, 0.99, 197):
        preds = (scores >= th).astype(int)
        m = compute_metrics(y_true, preds, scores)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = round(float(th), 4)
            best_m = m
    return best_th, best_m


def train_hybrid_classifier(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    use_gating: bool,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> tuple[HybridPairClassifier, float, dict]:
    name = "M5-Gated (Proposed)" if use_gating else "M5-Concat (Baseline)"
    print(f"\n{'='*70}\nHUẤN LUYỆN MÔ HÌNH: {name}\n{'='*70}")

    model = HybridPairClassifier(use_gating=use_gating, freeze_encoders=True, head_hidden=512, head_dropout=0.2).to(DEVICE)
    print(f"Tham số huấn luyện (Trainable): {model.trainable_param_count():,}")

    train_ds = TensorDataset(
        torch.tensor(train_data["text_a"], dtype=torch.float32),
        torch.tensor(train_data["graph_a"], dtype=torch.float32),
        torch.tensor(train_data["text_b"], dtype=torch.float32),
        torch.tensor(train_data["graph_b"], dtype=torch.float32),
        torch.tensor(train_data["labels"], dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(val_data["text_a"], dtype=torch.float32),
        torch.tensor(val_data["graph_a"], dtype=torch.float32),
        torch.tensor(val_data["text_b"], dtype=torch.float32),
        torch.tensor(val_data["graph_b"], dtype=torch.float32),
        torch.tensor(val_data["labels"], dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.head.parameters() if not use_gating else list(model.head.parameters()) + list(model.fusion_layer.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_weights = None
    best_opt_th = 0.5
    best_val_metrics = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ta, ga, tb, gb, y in train_loader:
            ta, ga, tb, gb, y = ta.to(DEVICE), ga.to(DEVICE), tb.to(DEVICE), gb.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model.forward_from_embeddings(ta, ga, tb, gb)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y)

        train_loss = total_loss / len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_scores, val_trues = [], []
        with torch.no_grad():
            for ta, ga, tb, gb, y in val_loader:
                ta, ga, tb, gb = ta.to(DEVICE), ga.to(DEVICE), tb.to(DEVICE), gb.to(DEVICE)
                logits = model.forward_from_embeddings(ta, ga, tb, gb)
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                val_scores.extend(probs)
                val_trues.extend(y.tolist())

        val_scores = np.array(val_scores)
        val_trues = np.array(val_trues, dtype=int)
        opt_th, val_m = find_best_threshold(val_trues, val_scores)

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_opt_th = opt_th
            best_val_metrics = val_m
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flag = " >>> BEST CHECKPOINT"
        else:
            flag = ""

        if ep % 5 == 0 or ep == 1 or "BEST" in flag:
            print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val F1: {val_m['f1']:.2f}% (th={opt_th:.3f}) | Val Acc: {val_m['accuracy']:.2f}%{flag}")

    # Nạp lại checkpoint tốt nhất trên validation
    model.load_state_dict(best_weights)
    print(f"=> Hoàn tất huấn luyện: Best Val F1 = {best_val_f1:.2f}% (ngưỡng tối ưu validation = {best_opt_th})")
    return model, best_opt_th, best_val_metrics


def evaluate_on_test(
    model: HybridPairClassifier,
    test_data: dict[str, np.ndarray],
    opt_th: float,
    batch_size: int = 64,
) -> tuple[dict, dict]:
    model.eval()
    test_ds = TensorDataset(
        torch.tensor(test_data["text_a"], dtype=torch.float32),
        torch.tensor(test_data["graph_a"], dtype=torch.float32),
        torch.tensor(test_data["text_b"], dtype=torch.float32),
        torch.tensor(test_data["graph_b"], dtype=torch.float32),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for ta, ga, tb, gb in test_loader:
            ta, ga, tb, gb = ta.to(DEVICE), ga.to(DEVICE), tb.to(DEVICE), gb.to(DEVICE)
            logits = model.forward_from_embeddings(ta, ga, tb, gb)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())

    scores = np.array(scores)
    y_true = test_data["labels"]

    # 1. Đánh giá tại ngưỡng cố định tìm được từ tập validation
    fixed_preds = (scores >= opt_th).astype(int)
    fixed_metrics = compute_metrics(y_true, fixed_preds, scores)
    fixed_metrics["threshold"] = opt_th

    # 2. Đánh giá tại ngưỡng tối ưu của chính tập test
    test_best_th, opt_metrics = find_best_threshold(y_true, scores)
    opt_metrics["threshold"] = test_best_th

    return fixed_metrics, opt_metrics


def main():
    print("=" * 70)
    print("THỰC NGHIỆM CẤP ĐỘ 2: HUẤN LUYỆN MÔ HÌNH LAI ĐẶC TRƯNG M5")
    print("=" * 70)

    # 1. Nạp dữ liệu các split
    train_pairs = load_pairs(TRAIN_FILE)
    val_pairs = load_pairs(VAL_FILE)
    test_pairs = load_pairs(TEST_FILE)
    print(f"Tổng số cặp: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    # 2. Khởi tạo và nạp pre-trained Encoders
    print("\nKhởi tạo bộ mã hóa từ checkpoint tốt nhất...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    m2_model = TransformerPairClassifier(hf_id="microsoft/graphcodebert-base", dropout=0.3).to(DEVICE)
    ckpt_m2 = torch.load(M2_CKPT, map_location=DEVICE, weights_only=False)
    m2_model.load_state_dict(ckpt_m2["model_state"])
    m2_model.eval()

    m4_model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2).to(DEVICE)
    ckpt_m4 = torch.load(M4_CKPT, map_location=DEVICE, weights_only=False)
    m4_model.load_state_dict(ckpt_m4["model_state"])
    m4_model.eval()

    # 3. Trích xuất đặc trưng và nạp cache
    train_data = extract_and_cache_embeddings(train_pairs, CACHE_TRAIN, m2_model, m4_model, tokenizer)
    val_data = extract_and_cache_embeddings(val_pairs, CACHE_VAL, m2_model, m4_model, tokenizer)
    test_data = extract_and_cache_embeddings(test_pairs, CACHE_TEST, m2_model, m4_model, tokenizer)

    # 4. Huấn luyện Mô hình 1: Concat-MLP Fusion (Baseline)
    m5_concat, th_concat, val_m_concat = train_hybrid_classifier(
        train_data, val_data, use_gating=False, epochs=25, lr=4e-4, weight_decay=1e-4
    )
    test_fixed_concat, test_opt_concat = evaluate_on_test(m5_concat, test_data, th_concat)
    torch.save(m5_concat.state_dict(), ROOT / "results" / "m5_hybrid_concat.pt")

    # 5. Huấn luyện Mô hình 2: Gated Cross-Modal Soft Fusion (Proposed)
    m5_gated, th_gated, val_m_gated = train_hybrid_classifier(
        train_data, val_data, use_gating=True, epochs=25, lr=3e-4, weight_decay=1e-4
    )
    test_fixed_gated, test_opt_gated = evaluate_on_test(m5_gated, test_data, th_gated)
    torch.save(m5_gated.state_dict(), ROOT / "results" / "m5_hybrid_gated.pt")

    # 6. So sánh tổng hợp
    results = {
        "m5_concat": {
            "val_opt_th": th_concat,
            "val_metrics": val_m_concat,
            "test_fixed": test_fixed_concat,
            "test_optimal": test_opt_concat,
        },
        "m5_gated": {
            "val_opt_th": th_gated,
            "val_metrics": val_m_gated,
            "test_fixed": test_fixed_gated,
            "test_optimal": test_opt_gated,
        },
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 85)
    print("BẢNG TỔNG KẾT SO SÁNH KẾT QUẢ KIỂM THỬ TRÊN 5.000 CẶP TEST (CẤP ĐỘ 2):")
    print("=" * 85)
    print(f"{'Mô hình':<35} | {'Ngưỡng tau':<10} | {'F1 (%)':<8} | {'Acc (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8} | {'TP':<5} | {'FP':<5} | {'FN':<5}")
    print("-" * 110)
    print(f"{'M4-EP (CPG-GNN)':<35} | {'0.30':<10} | {'79.84':<8} | {'77.50':<8} | {'72.33':<8} | {'89.08':<8} | {'2227':<5} | {'852':<5} | {'273':<5}")
    print(f"{'M2-DR (GraphCodeBERT)':<35} | {'0.52':<10} | {'96.31':<8} | {'96.36':<8} | {'97.66':<8} | {'95.00':<8} | {'2375':<5} | {'57':<5} | {'125':<5}")
    print(f"{'Hybrid Cấp độ 1 (Linear Blend)':<35} | {'0.405':<10} | {'96.40':<8} | {'96.44':<8} | {'97.58':<8} | {'95.24':<8} | {'2381':<5} | {'59':<5} | {'119':<5}")
    print(f"{'M5-Concat (Feature-Level)':<35} | {test_fixed_concat['threshold']:<10.3f} | {test_fixed_concat['f1']:<8.2f} | {test_fixed_concat['accuracy']:<8.2f} | {test_fixed_concat['precision']:<8.2f} | {test_fixed_concat['recall']:<8.2f} | {test_fixed_concat['tp']:<5} | {test_fixed_concat['fp']:<5} | {test_fixed_concat['fn']:<5}")
    print(f"{'M5-Gated (Feature-Level)':<35} | {test_fixed_gated['threshold']:<10.3f} | {test_fixed_gated['f1']:<8.2f} | {test_fixed_gated['accuracy']:<8.2f} | {test_fixed_gated['precision']:<8.2f} | {test_fixed_gated['recall']:<8.2f} | {test_fixed_gated['tp']:<5} | {test_fixed_gated['fp']:<5} | {test_fixed_gated['fn']:<5}")
    print("=" * 110)
    print(f"Đã lưu kết quả vào: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
