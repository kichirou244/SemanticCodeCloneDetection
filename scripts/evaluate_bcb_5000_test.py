#!/usr/bin/env python3
"""Danh gia 10 bien the mo hinh tren 5.000 cap Test BigCloneBench (CodeXGLUE)."""

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
PAIRS_CACHE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
RESULTS_FILE = ROOT / "results" / "bcb_test_5k_results.json"


def get_5k_test_pairs() -> list[dict]:
    if PAIRS_CACHE.exists():
        pairs = []
        with open(PAIRS_CACHE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs

    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_clone_detection_big_clone_bench", split="test")

    positives, negatives = [], []
    for item in ds:
        label = 1 if item["label"] else 0
        pair = {
            "id": item.get("id"),
            "func1": item["func1"].strip(),
            "func2": item["func2"].strip(),
            "label": label,
        }
        if label == 1:
            positives.append(pair)
        else:
            negatives.append(pair)
        if len(positives) >= 10000 and len(negatives) >= 10000:
            break

    rng = random.Random(42)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    all_pairs = positives[:2500] + negatives[:2500]
    rng.shuffle(all_pairs)

    PAIRS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_CACHE, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    return all_pairs


def load_saved_results() -> dict:
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_single_result(model_key: str, data: dict):
    res = load_saved_results()
    res[model_key] = data
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


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


def find_best_threshold(scores, labels, fixed_th=None) -> tuple[float, dict]:
    if fixed_th is not None:
        preds = [1 if s >= fixed_th else 0 for s in scores]
        return fixed_th, compute_metrics(labels, preds)

    best_th, best_f1, best_m = 0.5, -1.0, None
    for th in np.linspace(0.01, 0.99, 99):
        preds = [1 if s >= th else 0 for s in scores]
        m = compute_metrics(labels, preds)
        if m["f1"] > best_f1:
            best_f1, best_th, best_m = m["f1"], th, m
    return best_th, best_m


class MLPPairHead(nn.Module):
    def __init__(self, in_dim=2304, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h1, h2):
        pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)
        return self.net(pair).squeeze(-1)


def run_transformers(pairs, model_key, hf_id, mode, ckpt_path=None, fixed_th=None):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    labels = [p["label"] for p in pairs]
    codes_1 = [p["func1"] for p in pairs]
    codes_2 = [p["func2"] for p in pairs]
    scores = []
    bs = 64

    if mode == "raw":
        model = AutoModel.from_pretrained(hf_id).to(DEVICE)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                b1 = tokenizer(codes_1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                b2 = tokenizer(codes_2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                norm1 = F.normalize(model(**b1).last_hidden_state[:, 0, :], p=2, dim=1)
                norm2 = F.normalize(model(**b2).last_hidden_state[:, 0, :], p=2, dim=1)
                cos = (torch.sum(norm1 * norm2, dim=1) + 1.0) / 2.0
                scores.extend(cos.cpu().tolist())

    elif mode == "freeze":
        backbone = AutoModel.from_pretrained(hf_id).to(DEVICE)
        backbone.eval()
        head = MLPPairHead(in_dim=768 * 3, hidden_dim=768).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        head.net.load_state_dict(ckpt["head_state"])
        head.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                b1 = tokenizer(codes_1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                b2 = tokenizer(codes_2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                h1 = backbone(**b1).last_hidden_state[:, 0, :]
                h2 = backbone(**b2).last_hidden_state[:, 0, :]
                probs = torch.sigmoid(head(h1, h2)).cpu().tolist()
                scores.extend(probs if isinstance(probs, list) else [probs])

    elif mode == "full":
        from src.models.transformer_pair import TransformerPairClassifier
        model = TransformerPairClassifier(hf_id=hf_id, dropout=0.1).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                b1 = tokenizer(codes_1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                b2 = tokenizer(codes_2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
                scores.extend(probs if isinstance(probs, list) else [probs])

    th, m = find_best_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
    }
    save_single_result(model_key, res)
    return res


def run_astnn(pairs, model_key, mode, ckpt_path=None, fixed_th=None):
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees

    labels = [p["label"] for p in pairs]
    if mode == "raw":
        torch.manual_seed(42)
        model = ASTNNPairClassifier(hidden=200, dropout=0.0, n_stmt=45).to(DEVICE)
    else:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    scores = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            c1 = [compile_tree(extract_tree(p["func1"], language="java")) for p in chunk]
            c2 = [compile_tree(extract_tree(p["func2"], language="java")) for p in chunk]
            b1 = astnn_to_device(collate_trees(c1), DEVICE)
            b2 = astnn_to_device(collate_trees(c2), DEVICE)
            if mode == "raw":
                h1 = F.normalize(model.encode_side(b1), p=2, dim=1)
                h2 = F.normalize(model.encode_side(b2), p=2, dim=1)
                cos = (torch.sum(h1 * h2, dim=1) + 1.0) / 2.0
                scores.extend(cos.cpu().tolist())
            else:
                probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
                scores.extend(probs if isinstance(probs, list) else [probs])

    th, m = find_best_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
    }
    save_single_result(model_key, res)
    return res


def run_cpg(pairs, model_key, mode, ckpt_path=None, fixed_th=None):
    from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device as cpg_to_device
    from torch_geometric.data import Batch, Data
    from src.representations.ast.extract import extract_tree

    labels = [p["label"] for p in pairs]
    if mode == "raw":
        torch.manual_seed(42)
        model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.0).to(DEVICE)
    else:
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
        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            b1 = cpg_to_device(Batch.from_data_list([code_to_data(p["func1"]) for p in chunk]), DEVICE)
            b2 = cpg_to_device(Batch.from_data_list([code_to_data(p["func2"]) for p in chunk]), DEVICE)
            if mode == "raw":
                h1 = F.normalize(model.encoder(b1), p=2, dim=1)
                h2 = F.normalize(model.encoder(b2), p=2, dim=1)
                scores.extend(((torch.sum(h1 * h2, dim=1) + 1.0) / 2.0).cpu().tolist())
            else:
                probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
                scores.extend(probs if isinstance(probs, list) else [probs])

    th, m = find_best_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
    }
    save_single_result(model_key, res)
    return res


MODELS = {
    "M1_raw": lambda p: run_transformers(p, "M1_raw", "microsoft/codebert-base", "raw"),
    "M1_freeze": lambda p: run_transformers(p, "M1_freeze", "microsoft/codebert-base", "freeze", ckpt_path="results/M1_CodeBERT_Freeze/best_head.pt"),
    "M1_full": lambda p: run_transformers(p, "M1_full", "microsoft/codebert-base", "full", ckpt_path="results/exp_bcb_002/best_checkpoint.pt", fixed_th=0.88),
    "M2_raw": lambda p: run_transformers(p, "M2_raw", "microsoft/graphcodebert-base", "raw"),
    "M2_freeze": lambda p: run_transformers(p, "M2_freeze", "microsoft/graphcodebert-base", "freeze", ckpt_path="results/M2_GraphCodeBERT_Freeze/best_head.pt"),
    "M2_full": lambda p: run_transformers(p, "M2_full", "microsoft/graphcodebert-base", "full", ckpt_path="results/exp_bcb_003/best_checkpoint.pt", fixed_th=0.94),
    "M3_raw": lambda p: run_astnn(p, "M3_raw", "raw"),
    "M3_full": lambda p: run_astnn(p, "M3_full", "full", ckpt_path="results/exp_bcb_004/best_checkpoint.pt", fixed_th=0.52),
    "M4_raw": lambda p: run_cpg(p, "M4_raw", "raw"),
    "M4_full": lambda p: run_cpg(p, "M4_full", "full", ckpt_path="results/exp_bcb_005/best_checkpoint.pt", fixed_th=0.70),
}


def print_summary():
    res = load_saved_results()
    if not res:
        print("Chưa có kết quả.")
        return
    print(f"\n{'Mô hình':<20} | {'Ngưỡng':<7} | {'Đúng Clone':<12} | {'Đúng Non-clone':<15} | {'F1':<7} | {'Prec':<7} | {'Rec':<7} | {'Acc':<7}")
    print("-" * 92)
    for k in MODELS.keys():
        if k in res:
            v = res[k]
            print(f"{k:<20} | {v['th']:<7} | {v['tp']:>4}/2500     | {v['tn']:>4}/2500        | {v['f1']:<6.2f}% | {v['prec']:<6.2f}% | {v['rec']:<6.2f}% | {v['acc']:<6.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all", choices=["all", "summary"] + list(MODELS.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.model == "summary":
        print_summary()
        return

    pairs = get_5k_test_pairs()
    saved = load_saved_results()
    to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for m in to_run:
        if m in saved and not args.force:
            continue
        MODELS[m](pairs)
    print_summary()


if __name__ == "__main__":
    main()
