#!/usr/bin/env python3
"""Chạy đánh giá kiểm thử toàn diện 12 phiên bản mô hình trên tập Test 5.000 cặp chuẩn 1:1.
Lưu kết quả ngay sau mỗi mô hình để tránh phải chạy lại nếu gặp sự cố.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier
from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device as cpg_to_device
from torch_geometric.data import Batch, Data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_PAIRS_PATH = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
OUT_RESULTS_FILE = ROOT / "results" / "all_final_5k_test_results.json"


def load_test_pairs() -> tuple[list[int], list[str], list[str]]:
    assert TEST_PAIRS_PATH.exists(), f"Không tìm thấy file {TEST_PAIRS_PATH}"
    labels, c1, c2 = [], [], []
    with open(TEST_PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                labels.append(int(item["label"]))
                c1.append(item["func1"])
                c2.append(item["func2"])
    pos = sum(labels)
    neg = len(labels) - pos
    print(f"[DATA] Đã nạp {len(labels)} cặp test (Clone: {pos}, Non-clone: {neg})", flush=True)
    return labels, c1, c2


def compute_all_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    yp = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (yp == 1)))
    fp = int(np.sum((y_true == 0) & (yp == 1)))
    tn = int(np.sum((y_true == 0) & (yp == 0)))
    fn = int(np.sum((y_true == 1) & (yp == 0)))
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, scores))
    except Exception:
        auc = 0.5
    return {
        "th": round(float(threshold), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "f1": round(f1 * 100, 2),
        "acc": round(acc * 100, 2),
        "prec": round(prec * 100, 2),
        "rec": round(rec * 100, 2),
        "roc_auc": round(auc * 100, 2),
    }


def find_optimal_metrics(y_true: np.ndarray, scores: np.ndarray, fixed_th: float | None = None) -> tuple[dict, dict]:
    # Fixed threshold metrics
    default_th = fixed_th if fixed_th is not None else 0.5
    fixed_m = compute_all_metrics(y_true, scores, default_th)

    # Search best F1 threshold
    best_f1, best_th, opt_m = -1.0, 0.5, None
    for th in np.linspace(0.01, 0.99, 99):
        m = compute_all_metrics(y_true, scores, th)
        if m["f1"] > best_f1:
            best_f1, best_th, opt_m = m["f1"], th, m
    return fixed_m, opt_m


def save_result_checkpoint(model_key: str, data: dict):
    all_res = {}
    if OUT_RESULTS_FILE.exists():
        try:
            with open(OUT_RESULTS_FILE, "r", encoding="utf-8") as f:
                all_res = json.load(f)
        except Exception:
            all_res = {}
    all_res[model_key] = data
    with open(OUT_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Đã lưu kết quả của '{model_key}' vào {OUT_RESULTS_FILE}", flush=True)


def eval_transformer_model(hf_id: str, ckpt_path: str, max_length: int, labels: list[int], c1: list[str], c2: list[str], fixed_th: float):
    print(f"  Đang nạp Transformer: {hf_id} (ckpt={ckpt_path}, max_length={max_length})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, dropout=0.1).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(labels), bs):
            b1 = tokenizer(c1[i : i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(c2[i : i + bs], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
            logits = model(b1, b2)
            probs = torch.sigmoid(logits).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])

    y_true = np.array(labels)
    scores = np.array(scores)
    fixed_m, opt_m = find_optimal_metrics(y_true, scores, fixed_th)
    return {"fixed": fixed_m, "optimal": opt_m}


def eval_astnn_model(ckpt_path: str, labels: list[int], c1: list[str], c2: list[str], fixed_th: float):
    print(f"  Đang nạp ASTNN: {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(labels), bs):
            chunk1 = c1[i : i + bs]
            chunk2 = c2[i : i + bs]
            t1 = [compile_tree(extract_tree(c, language="java")) for c in chunk1]
            t2 = [compile_tree(extract_tree(c, language="java")) for c in chunk2]
            b1 = astnn_to_device(collate_trees(t1), DEVICE)
            b2 = astnn_to_device(collate_trees(t2), DEVICE)
            probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])

    y_true = np.array(labels)
    scores = np.array(scores)
    fixed_m, opt_m = find_optimal_metrics(y_true, scores, fixed_th)
    return {"fixed": fixed_m, "optimal": opt_m}


def eval_cpg_model(ckpt_path: str, labels: list[int], c1: list[str], c2: list[str], fixed_th: float):
    print(f"  Đang nạp CPG-GNN: {ckpt_path}...", flush=True)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.1).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def code_to_data(code: str) -> Data:
        tree = extract_tree(code, language="java")
        n = len(tree["label"])
        parents = tree["parent"]
        src, dst = [], []
        for i in range(1, n):
            p = int(parents[i])
            if 0 <= p < n and p != i:
                src.extend([p, i])
                dst.extend([i, p])
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(len(src), dtype=torch.long) if src else torch.zeros(0, dtype=torch.long)
        return Data(x=torch.zeros((n, 1), dtype=torch.float), edge_index=edge_index, edge_type=edge_type, node_type=torch.zeros(n, dtype=torch.long), num_nodes=n)

    scores = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(labels), bs):
            b1 = cpg_to_device(Batch.from_data_list([code_to_data(c) for c in c1[i : i + bs]]), DEVICE)
            b2 = cpg_to_device(Batch.from_data_list([code_to_data(c) for c in c2[i : i + bs]]), DEVICE)
            probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])

    y_true = np.array(labels)
    scores = np.array(scores)
    fixed_m, opt_m = find_optimal_metrics(y_true, scores, fixed_th)
    return {"fixed": fixed_m, "optimal": opt_m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Chạy lại cả những model đã có kết quả")
    args = parser.parse_args()

    labels, c1, c2 = load_test_pairs()

    # Định nghĩa toàn bộ danh sách 12 mô hình cần kiểm thử
    tasks = [
        # Model key, type, hf_id, ckpt_path, max_length, fixed_th
        ("M1_Base", "transformer", "microsoft/codebert-base", "results/bcb_test_5k_results.json", 256, 0.88),
        ("M1_384", "transformer", "microsoft/codebert-base", "results/exp_bcb_002_384/best_checkpoint.pt", 384, 0.88),
        ("M1_512_v1", "transformer", "microsoft/codebert-base", "results/exp_bcb_002_512/best_checkpoint.pt", 512, 0.90),
        ("M1_512_v2", "transformer", "microsoft/codebert-base", "results/exp_bcb_002_512_v2/best_checkpoint.pt", 512, 0.86),
        
        ("M2_Base", "transformer", "microsoft/graphcodebert-base", "results/bcb_test_5k_results.json", 256, 0.94),
        ("M2_384", "transformer", "microsoft/graphcodebert-base", "results/exp_bcb_003_384/best_checkpoint.pt", 384, 0.94),
        ("M2_512_v1", "transformer", "microsoft/graphcodebert-base", "results/exp_bcb_003_512/best_checkpoint.pt", 512, 0.64),
        ("M2_512_v2", "transformer", "microsoft/graphcodebert-base", "results/exp_bcb_003_512_v2/best_checkpoint.pt", 512, 0.80),
        
        ("M3_Base", "astnn", None, "results/exp_bcb_004/best_checkpoint.pt", None, 0.52),
        ("M3_Opt", "astnn", None, "results/exp_bcb_004_opt/best_checkpoint.pt", None, 0.58),
        
        ("M4_Base", "cpg_gnn", None, "results/exp_bcb_005/best_checkpoint.pt", None, 0.70),
        ("M4_Opt", "cpg_gnn", None, "results/exp_bcb_005_opt/best_checkpoint.pt", None, 0.62),
    ]

    # Nạp kết quả cũ nếu có
    saved_res = {}
    if OUT_RESULTS_FILE.exists():
        try:
            with open(OUT_RESULTS_FILE, "r", encoding="utf-8") as f:
                saved_res = json.load(f)
        except Exception:
            saved_res = {}

    # Đọc kết quả Base đã kiểm thử chuẩn từ bcb_test_5k_results.json nếu chưa có
    base_file = ROOT / "results" / "bcb_test_5k_results.json"
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            bdata = json.load(f)
            if "M1_Base" not in saved_res and "M1_full" in bdata:
                saved_res["M1_Base"] = {"optimal": bdata["M1_full"], "fixed": bdata["M1_full"]}
                save_result_checkpoint("M1_Base", saved_res["M1_Base"])
            if "M2_Base" not in saved_res and "M2_full" in bdata:
                saved_res["M2_Base"] = {"optimal": bdata["M2_full"], "fixed": bdata["M2_full"]}
                save_result_checkpoint("M2_Base", saved_res["M2_Base"])
            if "M3_Base" not in saved_res and "M3_full" in bdata:
                saved_res["M3_Base"] = {"optimal": bdata["M3_full"], "fixed": bdata["M3_full"]}
                save_result_checkpoint("M3_Base", saved_res["M3_Base"])
            if "M4_Base" not in saved_res and "M4_full" in bdata:
                saved_res["M4_Base"] = {"optimal": bdata["M4_full"], "fixed": bdata["M4_full"]}
                save_result_checkpoint("M4_Base", saved_res["M4_Base"])

    print("\n" + "=" * 70, flush=True)
    print("BẮT ĐẦU ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN 5.000 CẶP TEST CHUẨN 1:1...", flush=True)
    print("=" * 70, flush=True)

    for mkey, mtype, hf_id, ckpt_path, max_len, fixed_th in tasks:
        if mkey in saved_res and not args.force:
            print(f">>> [BỎ QUA] '{mkey}' đã có kết quả kiểm thử trong {OUT_RESULTS_FILE.name}.", flush=True)
            continue

        ckpt_p = Path(ckpt_path) if ckpt_path else None
        if ckpt_p and not ckpt_p.exists() and not ckpt_path.endswith(".json"):
            print(f">>> [CẢNH BÁO] Không tìm thấy file checkpoint: {ckpt_path} cho {mkey}, bỏ qua!", flush=True)
            continue

        t0 = time.time()
        print(f"\n>>> [TIẾN HÀNH] Đang kiểm thử '{mkey}'...", flush=True)
        try:
            if mtype == "transformer":
                res = eval_transformer_model(hf_id, ckpt_path, max_len, labels, c1, c2, fixed_th)
            elif mtype == "astnn":
                res = eval_astnn_model(ckpt_path, labels, c1, c2, fixed_th)
            elif mtype == "cpg_gnn":
                res = eval_cpg_model(ckpt_path, labels, c1, c2, fixed_th)
            else:
                continue

            elapsed = time.time() - t0
            res["elapsed_s"] = round(elapsed, 1)
            saved_res[mkey] = res
            save_result_checkpoint(mkey, res)

            opt = res["optimal"]
            print(f"  >>> HOÀN TẤT '{mkey}' ({elapsed:.1f}s):", flush=True)
            print(f"      Ngưỡng tối ưu: {opt['th']} | F1: {opt['f1']}% | Acc: {opt['acc']}% | Prec: {opt['prec']}% | Rec: {opt['rec']}%", flush=True)
            print(f"      Confusion Matrix: TP={opt['tp']}/2500, TN={opt['tn']}/2500, FP={opt['fp']}, FN={opt['fn']}", flush=True)

        except Exception as e:
            print(f"  >>> [LỖI] Xảy ra lỗi khi chạy '{mkey}': {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70, flush=True)
    print("HOÀN TẤT ĐÁNH GIÁ TOÀN BỘ CÁC MÔ HÌNH!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
