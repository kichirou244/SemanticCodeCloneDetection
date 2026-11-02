#!/usr/bin/env python3
"""Chon nguong toi uu tren tap Validation (BigCloneBench), sau do ap dung nguong do
de danh gia lai ca tap Test (5.000 mau) va tap Train (500 mau).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_CACHE = ROOT / "results" / "bcb_1k_val_pairs.jsonl"
TEST_CACHE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
TRAIN_CACHE = ROOT / "results" / "bcb_500_train_pairs.jsonl"
OUTPUT_FILE = ROOT / "results" / "val_tuned_thresholds_results.json"


def get_balanced_val_pairs(n_pos=500, n_neg=500) -> list[dict]:
    if VAL_CACHE.exists():
        pairs = []
        with open(VAL_CACHE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs

    print("Dang trich xuat 1.000 mau can bang tu tap Validation...")
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench", split="validation")

    positives, negatives = [], []
    for item in ds:
        label = 1 if item["label"] else 0
        pair = {
            "id": item.get("id"),
            "func1": item["func1"].strip(),
            "func2": item["func2"].strip(),
            "label": label,
        }
        if label == 1 and len(positives) < n_pos * 3:
            positives.append(pair)
        elif label == 0 and len(negatives) < n_neg * 3:
            negatives.append(pair)
        if len(positives) >= n_pos * 3 and len(negatives) >= n_neg * 3:
            break

    rng = random.Random(42)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    all_pairs = positives[:n_pos] + negatives[:n_neg]
    rng.shuffle(all_pairs)

    VAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(VAL_CACHE, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    return all_pairs


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc, "prec": prec, "rec": rec, "f1": f1}


def find_best_threshold(scores, labels) -> tuple[float, float]:
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.01, 0.99, 99):
        preds = [1 if s >= th else 0 for s in scores]
        m = compute_metrics(labels, preds)
        if m["f1"] > best_f1:
            best_f1, best_th = m["f1"], th
    return round(float(best_th), 4), best_f1


def eval_with_threshold(scores, labels, th: float) -> dict:
    preds = [1 if s >= th else 0 for s in scores]
    m = compute_metrics(labels, preds)
    return {
        "th": th,
        "f1": round(m["f1"] * 100, 2),
        "acc": round(m["acc"] * 100, 2),
        "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
    }


def predict_transformer(model, tokenizer, pairs, bs=64) -> list[float]:
    model.eval()
    codes_1 = [p["func1"] for p in pairs]
    codes_2 = [p["func2"] for p in pairs]
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            b1 = tokenizer(codes_1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(codes_2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])
    return scores


def predict_astnn(model, pairs, bs=64) -> list[float]:
    from src.models.astnn.model import to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees

    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            c1 = [compile_tree(extract_tree(p["func1"], language="java")) for p in chunk]
            c2 = [compile_tree(extract_tree(p["func2"], language="java")) for p in chunk]
            b1 = astnn_to_device(collate_trees(c1), DEVICE)
            b2 = astnn_to_device(collate_trees(c2), DEVICE)
            probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])
    return scores


def predict_cpg(model, pairs, bs=64) -> list[float]:
    from src.models.cpg_gnn.model import to_device as cpg_to_device
    from src.representations.ast.extract import extract_tree
    from torch_geometric.data import Batch, Data

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

    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            b1 = cpg_to_device(Batch.from_data_list([code_to_data(p["func1"]) for p in chunk]), DEVICE)
            b2 = cpg_to_device(Batch.from_data_list([code_to_data(p["func2"]) for p in chunk]), DEVICE)
            probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
            scores.extend(probs if isinstance(probs, list) else [probs])
    return scores


def main():
    print(f"Device: {DEVICE}")
    val_pairs = get_balanced_val_pairs(500, 500)
    test_pairs = load_pairs(TEST_CACHE)
    train_pairs = load_pairs(TRAIN_CACHE)

    val_labels = [p["label"] for p in val_pairs]
    test_labels = [p["label"] for p in test_pairs]
    train_labels = [p["label"] for p in train_pairs]

    print(f"Loaded: Val={len(val_pairs)}, Test={len(test_pairs)}, Train_500={len(train_pairs)}")

    results = {}

    # 1. M1 - CodeBERT Full
    print("\n[1/4] Dang xu ly M1: CodeBERT...")
    from src.models.transformer_pair import TransformerPairClassifier
    tok1 = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    m1 = TransformerPairClassifier(hf_id="microsoft/codebert-base", dropout=0.1).to(DEVICE)
    ckpt1 = torch.load("results/exp_bcb_002/best_checkpoint.pt", map_location=DEVICE, weights_only=False)
    m1.load_state_dict(ckpt1["model_state"])
    val_s1 = predict_transformer(m1, tok1, val_pairs)
    th1, best_val_f1 = find_best_threshold(val_s1, val_labels)
    print(f"  -> Best Val Threshold = {th1} (Val F1 = {best_val_f1*100:.2f}%)")
    test_s1 = predict_transformer(m1, tok1, test_pairs)
    train_s1 = predict_transformer(m1, tok1, train_pairs)
    results["M1 - CodeBERT"] = {
        "th": th1,
        "val_f1": round(best_val_f1 * 100, 2),
        "test": eval_with_threshold(test_s1, test_labels, th1),
        "train": eval_with_threshold(train_s1, train_labels, th1),
    }

    # 2. M2 - GraphCodeBERT Full
    print("\n[2/4] Dang xu ly M2: GraphCodeBERT...")
    tok2 = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    m2 = TransformerPairClassifier(hf_id="microsoft/graphcodebert-base", dropout=0.1).to(DEVICE)
    ckpt2 = torch.load("results/exp_bcb_003/best_checkpoint.pt", map_location=DEVICE, weights_only=False)
    m2.load_state_dict(ckpt2["model_state"])
    val_s2 = predict_transformer(m2, tok2, val_pairs)
    th2, best_val_f1_2 = find_best_threshold(val_s2, val_labels)
    print(f"  -> Best Val Threshold = {th2} (Val F1 = {best_val_f1_2*100:.2f}%)")
    test_s2 = predict_transformer(m2, tok2, test_pairs)
    train_s2 = predict_transformer(m2, tok2, train_pairs)
    results["M2 - GraphCodeBERT"] = {
        "th": th2,
        "val_f1": round(best_val_f1_2 * 100, 2),
        "test": eval_with_threshold(test_s2, test_labels, th2),
        "train": eval_with_threshold(train_s2, train_labels, th2),
    }

    # 3. M3 - ASTNN Full
    print("\n[3/4] Dang xu ly M3: ASTNN...")
    from src.models.astnn.model import ASTNNPairClassifier
    m3 = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
    ckpt3 = torch.load("results/exp_bcb_004/best_checkpoint.pt", map_location=DEVICE, weights_only=False)
    m3.load_state_dict(ckpt3["model_state"])
    val_s3 = predict_astnn(m3, val_pairs)
    th3, best_val_f1_3 = find_best_threshold(val_s3, val_labels)
    print(f"  -> Best Val Threshold = {th3} (Val F1 = {best_val_f1_3*100:.2f}%)")
    test_s3 = predict_astnn(m3, test_pairs)
    train_s3 = predict_astnn(m3, train_pairs)
    results["M3 - ASTNN"] = {
        "th": th3,
        "val_f1": round(best_val_f1_3 * 100, 2),
        "test": eval_with_threshold(test_s3, test_labels, th3),
        "train": eval_with_threshold(train_s3, train_labels, th3),
    }

    # 4. M4 - CPG-GNN Full
    print("\n[4/4] Dang xu ly M4: CPG-GNN...")
    from src.models.cpg_gnn.model import CPGGNNPairClassifier
    m4 = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.1).to(DEVICE)
    ckpt4 = torch.load("results/exp_bcb_005/best_checkpoint.pt", map_location=DEVICE, weights_only=False)
    m4.load_state_dict(ckpt4["model_state"])
    val_s4 = predict_cpg(m4, val_pairs)
    th4, best_val_f1_4 = find_best_threshold(val_s4, val_labels)
    print(f"  -> Best Val Threshold = {th4} (Val F1 = {best_val_f1_4*100:.2f}%)")
    test_s4 = predict_cpg(m4, test_pairs)
    train_s4 = predict_cpg(m4, train_pairs)
    results["M4 - CPG-GNN"] = {
        "th": th4,
        "val_f1": round(best_val_f1_4 * 100, 2),
        "test": eval_with_threshold(test_s4, test_labels, th4),
        "train": eval_with_threshold(train_s4, train_labels, th4),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDa luu ket qua vao: {OUTPUT_FILE}")

    # In bang tong hop
    print("\n" + "="*80)
    print(f"{'Mô hình':<20} | {'Ngưỡng (Val)':<12} | {'Train F1':<9} | {'Test F1':<9} | {'Delta F1':<9} | {'Train Acc':<9} | {'Test Acc':<9} | {'Delta Acc':<9}")
    print("-"*95)
    for name, r in results.items():
        th = r["th"]
        tr_f1 = r["train"]["f1"]
        te_f1 = r["test"]["f1"]
        d_f1 = tr_f1 - te_f1
        tr_acc = r["train"]["acc"]
        te_acc = r["test"]["acc"]
        d_acc = tr_acc - te_acc
        print(f"{name:<20} | {th:<12.2f} | {tr_f1:<8.2f}% | {te_f1:<8.2f}% | {d_f1:>+7.2f}% | {tr_acc:<8.2f}% | {te_acc:<8.2f}% | {d_acc:>+7.2f}%")


if __name__ == "__main__":
    main()
