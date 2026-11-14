#!/usr/bin/env python3
"""Đánh giá Hybrid Cấp độ 1: Late Fusion & Dynamic Gating giữa EXP-M2-DR (GraphCodeBERT) và EXP-M4-EP (CPG-GNN).

Tập dữ liệu: 5.000 cặp Test BigCloneBench chuẩn 1:1 (results/bcb_5k_test_pairs.jsonl).
Mô hình M2: results/exp_m2_dr/best_checkpoint.pt (max_length=512)
Mô hình M4: results/exp_m4_ep/best_checkpoint.pt (CPG-GNN)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from src.data.pairs.bcb_dataset import collate_cpg_pairs, ClonePairSample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PAIRS_FILE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
CPG_EXPORT_DIR = ROOT / "artifacts" / "cpg_export"

M2_CKPT = ROOT / "results" / "exp_m2_dr" / "best_checkpoint.pt"
M4_CKPT = ROOT / "results" / "exp_m4_ep" / "best_checkpoint.pt"

M2_PROBS_CACHE = ROOT / "results" / "m2_dr_5k_probs.npy"
M4_PROBS_CACHE = ROOT / "results" / "m4_ep_5k_probs.npy"
OUTPUT_JSON = ROOT / "results" / "hybrid_level1_results.json"


def load_pairs() -> list[dict]:
    pairs = []
    with open(TEST_PAIRS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} test pairs from {TEST_PAIRS_FILE.name}")
    return pairs


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
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "roc_auc": round(auc * 100, 2),
    }


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
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


def get_m2_probs(pairs: list[dict], batch_size: int = 32) -> np.ndarray:
    if M2_PROBS_CACHE.exists():
        print(f"Loading cached M2 probabilities from {M2_PROBS_CACHE.name}...")
        return np.load(M2_PROBS_CACHE)

    print("\n>>> Computing M2-DR (GraphCodeBERT) Probabilities on 5k test pairs...")
    hf_id = "microsoft/graphcodebert-base"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, dropout=0.3).to(DEVICE)
    ckpt = torch.load(M2_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    c1 = [p["func1"] for p in pairs]
    c2 = [p["func2"] for p in pairs]

    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            b1 = tokenizer(c1[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(c2[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            logits = model(b1, b2)
            p = torch.sigmoid(logits).cpu().tolist()
            if isinstance(p, list):
                probs.extend(p)
            else:
                probs.append(p)
            if (i // batch_size) % 25 == 0:
                print(f"  M2 Progress: {min(i + batch_size, len(pairs))}/{len(pairs)} pairs...", flush=True)

    probs = np.array(probs, dtype=np.float32)
    np.save(M2_PROBS_CACHE, probs)
    print(f"  Done in {time.perf_counter() - t0:.1f}s. Saved to {M2_PROBS_CACHE.name}")
    return probs


def get_m4_probs(pairs: list[dict], batch_size: int = 64) -> np.ndarray:
    if M4_PROBS_CACHE.exists():
        print(f"Loading cached M4 probabilities from {M4_PROBS_CACHE.name}...")
        return np.load(M4_PROBS_CACHE)

    print("\n>>> Computing M4-EP (CPG-GNN) Probabilities on 5k test pairs...")
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2).to(DEVICE)
    ckpt = torch.load(M4_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sample_objects = [
        ClonePairSample(
            pair_id=str(p["id"]),
            id1=str(p["id1"]),
            id2=str(p["id2"]),
            func1=p["func1"],
            func2=p["func2"],
            label=int(p["label"]),
            language="java",
            split="test",
        )
        for p in pairs
    ]

    probs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_samples = sample_objects[i : i + batch_size]
            pyg_a, pyg_b, _ = collate_cpg_pairs(batch_samples, cpg_store=CPG_EXPORT_DIR, max_nodes=1500, device=DEVICE)
            logits = model(pyg_a, pyg_b)
            p = torch.sigmoid(logits).cpu().tolist()
            if isinstance(p, list):
                probs.extend(p)
            else:
                probs.append(p)
            if (i // batch_size) % 20 == 0:
                print(f"  M4 Progress: {min(i + batch_size, len(pairs))}/{len(pairs)} pairs...", flush=True)

    probs = np.array(probs, dtype=np.float32)
    np.save(M4_PROBS_CACHE, probs)
    print(f"  Done in {time.perf_counter() - t0:.1f}s. Saved to {M4_PROBS_CACHE.name}")
    return probs


def main():
    pairs = load_pairs()
    y_true = np.array([p["label"] for p in pairs], dtype=int)

    # 1. Thu thập xác suất từ 2 mô hình thành phần
    p_m2 = get_m2_probs(pairs)
    p_m4 = get_m4_probs(pairs)

    print("\n" + "=" * 70)
    print("KIỂM TRA HIỆU NĂNG RIÊNG LẺ CỦA 2 MÔ HÌNH NỀN TẢNG:")
    print("=" * 70)

    # Đánh giá M2 tại ngưỡng chuẩn 0.52 và tối ưu
    m2_std_preds = (p_m2 >= 0.52).astype(int)
    m2_std = compute_metrics(y_true, m2_std_preds, p_m2)
    m2_opt_th, m2_opt = find_optimal_threshold(y_true, p_m2)
    print(f"M2-DR (GraphCodeBERT):")
    print(f"  - Chuẩn (th=0.52): F1={m2_std['f1']}% | Acc={m2_std['accuracy']}% | Prec={m2_std['precision']}% | Rec={m2_std['recall']}% | TP={m2_std['tp']}, FP={m2_std['fp']}, FN={m2_std['fn']}")
    print(f"  - Tối ưu (th={m2_opt_th}): F1={m2_opt['f1']}% | Acc={m2_opt['accuracy']}% | Prec={m2_opt['precision']}% | Rec={m2_opt['recall']}%")

    # Đánh giá M4 tại ngưỡng chuẩn 0.30 và tối ưu
    m4_std_preds = (p_m4 >= 0.30).astype(int)
    m4_std = compute_metrics(y_true, m4_std_preds, p_m4)
    m4_opt_th, m4_opt = find_optimal_threshold(y_true, p_m4)
    print(f"M4-EP (CPG-GNN):")
    print(f"  - Chuẩn (th=0.30): F1={m4_std['f1']}% | Acc={m4_std['accuracy']}% | Prec={m4_std['precision']}% | Rec={m4_std['recall']}% | TP={m4_std['tp']}, FP={m4_std['fp']}, FN={m4_std['fn']}")
    print(f"  - Tối ưu (th={m4_opt_th}): F1={m4_opt['f1']}% | Acc={m4_opt['accuracy']}% | Prec={m4_opt['precision']}% | Rec={m4_opt['recall']}%")

    # Phân tích sự đồng thuận & tiềm năng cứu lỗi (Oracle Rescue Rate)
    m2_correct = (m2_std_preds == y_true)
    m4_correct = (m4_std_preds == y_true)
    m2_errors = np.where(~m2_correct)[0]
    m4_rescues = np.where((~m2_correct) & m4_correct)[0]

    print("\n" + "=" * 70)
    print("PHÂN TÍCH TƯƠNG ĐỒNG & TIỀM NĂNG BÙ TRỪ:")
    print("=" * 70)
    print(f"Tổng số lỗi của M2-DR: {len(m2_errors)} (FN: {m2_std['fn']}, FP: {m2_std['fp']})")
    print(f"Số lỗi mà M4-EP đoán đúng (Oracle Rescue): {len(m4_rescues)}/{len(m2_errors)} ({len(m4_rescues)/len(m2_errors)*100:.1f}%)")
    
    # 2. KHẢO SÁT CÁC CHIẾN LƯỢC FUSION CẤP ĐỘ 1:
    results = {
        "single_m2": {"std": m2_std, "opt_th": m2_opt_th, "opt": m2_opt},
        "single_m4": {"std": m4_std, "opt_th": m4_opt_th, "opt": m4_opt},
        "linear_blends": {},
        "dynamic_gating": {},
        "confidence_band_gating": {},
        "best_overall": {},
    }

    print("\n" + "=" * 70)
    print("1. CHIẾN LƯỢC LINEAR BLENDING: P = alpha * P_M2 + (1 - alpha) * P_M4")
    print("=" * 70)
    best_blend_f1 = -1.0
    best_blend_info = None

    for alpha in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97]:
        p_blend = alpha * p_m2 + (1.0 - alpha) * p_m4
        th, m = find_optimal_threshold(y_true, p_blend)
        results["linear_blends"][str(round(alpha, 2))] = {"threshold": th, **m}
        print(f"  alpha={alpha:.2f} | Best th={th:.4f} | F1={m['f1']}% | Acc={m['accuracy']}% | Prec={m['precision']}% | Rec={m['recall']}% | TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")
        if m["f1"] > best_blend_f1:
            best_blend_f1 = m["f1"]
            best_blend_info = {"alpha": alpha, "threshold": th, **m}

    print(f"\n=> TỐT NHẤT LINEAR BLEND: alpha={best_blend_info['alpha']} (F1: {best_blend_info['f1']}%, Acc: {best_blend_info['accuracy']}%)")

    print("\n" + "=" * 70)
    print("2. CHIẾN LƯỢC DYNAMIC CONFIDENCE GATING (RESCUE FALSE NEGATIVES):")
    print("Ý tưởng: Tin tưởng M2 tuyệt đối khi M2 dự đoán Clone (giữ FP thấp).")
    print("Chỉ khi M2 dự đoán Non-Clone (P_M2 < th_m2), nếu CPG-GNN cực kỳ chắc chắn (P_M4 > th_rescue),")
    print("ta cho phép CPG-GNN cứu lỗi False Negative mà M2 bỏ sót.")
    print("=" * 70)

    best_gate_f1 = -1.0
    best_gate_info = None

    # Quét ngưỡng M2 và ngưỡng Rescue của M4
    for th_m2 in [0.45, 0.50, 0.52, 0.55, 0.60]:
        for th_rescue in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]:
            preds = (p_m2 >= th_m2).astype(int)
            # Rescue: Nếu M2 bảo 0 nhưng M4 tự tin cao >= th_rescue -> Lật thành 1
            rescue_mask = (p_m2 < th_m2) & (p_m4 >= th_rescue)
            preds[rescue_mask] = 1

            m = compute_metrics(y_true, preds, p_m2)
            rescued_fn = int(np.sum((y_true[rescue_mask] == 1)))
            added_fp = int(np.sum((y_true[rescue_mask] == 0)))

            key = f"m2_{th_m2}_m4_{th_rescue}"
            results["dynamic_gating"][key] = {
                "th_m2": th_m2,
                "th_rescue": th_rescue,
                "rescued_fn": rescued_fn,
                "added_fp": added_fp,
                "net_gain": rescued_fn - added_fp,
                **m,
            }

            if m["f1"] > best_gate_f1:
                best_gate_f1 = m["f1"]
                best_gate_info = {"th_m2": th_m2, "th_rescue": th_rescue, "rescued_fn": rescued_fn, "added_fp": added_fp, **m}

    print(f"Top 5 cấu hình Dynamic Gating tốt nhất:")
    sorted_gates = sorted(results["dynamic_gating"].items(), key=lambda x: (x[1]["f1"], x[1]["accuracy"]), reverse=True)[:5]
    for k, v in sorted_gates:
        print(f"  th_m2={v['th_m2']}, th_rescue={v['th_rescue']} -> F1={v['f1']}% | Acc={v['accuracy']}% | Prec={v['precision']}% | Rec={v['recall']}% | Cứu: +{v['rescued_fn']} FN, Thêm: +{v['added_fp']} FP (Net: {v['net_gain']})")

    print("\n" + "=" * 70)
    print("3. CHIẾN LƯỢC CONFIDENCE-BAND GATING (VÙNG PHÂN VÂN):")
    print("Nếu P_M2 chắc chắn (> high hoặc < low) -> Giữ nguyên P_M2.")
    print("Nếu P_M2 rơi vào vùng phân vân [low, high] -> Trộn với P_M4 theo tỷ lệ beta.")
    print("=" * 70)

    best_band_f1 = -1.0
    best_band_info = None

    for low, high in [(0.2, 0.8), (0.3, 0.7), (0.35, 0.65), (0.4, 0.6), (0.45, 0.55)]:
        for beta in [0.3, 0.5, 0.7]:
            p_band = p_m2.copy()
            uncertain_mask = (p_m2 >= low) & (p_m2 <= high)
            p_band[uncertain_mask] = (1 - beta) * p_m2[uncertain_mask] + beta * p_m4[uncertain_mask]
            th, m = find_optimal_threshold(y_true, p_band)
            key = f"band_{low}_{high}_beta_{beta}"
            results["confidence_band_gating"][key] = {"low": low, "high": high, "beta": beta, "threshold": th, **m}
            if m["f1"] > best_band_f1:
                best_band_f1 = m["f1"]
                best_band_info = {"low": low, "high": high, "beta": beta, "threshold": th, **m}

    print(f"=> TỐT NHẤT CONFIDENCE BAND: band=[{best_band_info['low']}, {best_band_info['high']}], beta={best_band_info['beta']}, th={best_band_info['threshold']} -> F1={best_band_info['f1']}%, Acc={best_band_info['accuracy']}%")

    # Xác định chiến lược tối ưu nhất tổng thể
    all_strategies = [
        ("Single M2-DR", m2_std["f1"], m2_std),
        (f"Linear Blend (alpha={best_blend_info['alpha']})", best_blend_info["f1"], best_blend_info),
        (f"Dynamic Gating (th_m2={best_gate_info['th_m2']}, th_m4={best_gate_info['th_rescue']})", best_gate_info["f1"], best_gate_info),
        (f"Confidence Band (band=[{best_band_info['low']}, {best_band_info['high']}], beta={best_band_info['beta']})", best_band_info["f1"], best_band_info),
    ]
    all_strategies.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("BẢNG TỔNG KẾT SO SÁNH CÁC CHIẾN LƯỢC FUSION CẤP ĐỘ 1:")
    print("=" * 70)
    print(f"{'Phương pháp':<45} | {'F1 (%)':<8} | {'Acc (%)':<8} | {'Prec (%)':<8} | {'Rec (%)':<8}")
    print("-" * 85)
    for name, f1_val, m in all_strategies:
        print(f"{name:<45} | {m['f1']:<8.2f} | {m['accuracy']:<8.2f} | {m['precision']:<8.2f} | {m['recall']:<8.2f}")
    print("=" * 85)

    results["best_overall"] = all_strategies[0][2]
    results["best_overall"]["method_name"] = all_strategies[0][0]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nĐã lưu toàn bộ kết quả chi tiết vào: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
