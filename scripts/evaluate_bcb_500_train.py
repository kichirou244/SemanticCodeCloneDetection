#!/usr/bin/env python3
"""Danh gia cac bien the mo hinh tren 500 cap mau ngau nhien trich xuat tu tap Train (BigCloneBench)
de danh gia overfit bang cach doi chieu truc tiep voi ket qua tap Test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
TRAIN_PAIRS_FILE = ROOT / "results" / "bcb_500_train_pairs.jsonl"
RESULTS_FILE = ROOT / "results" / "bcb_train_500_results.json"
TEST_RESULTS_FILE = ROOT / "results" / "bcb_test_5k_results.json"


def load_500_train_pairs() -> list[dict]:
    if not TRAIN_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Khong tim thay file {TRAIN_PAIRS_FILE}")
    pairs = []
    with open(TRAIN_PAIRS_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def load_test_thresholds() -> dict[str, float]:
    """Lay lai chinh xac nguong danh gia da ap dung tren tap Test (5.000 mau)."""
    defaults = {
        "M1_raw": 0.01,
        "M1_freeze": 0.91,
        "M1_full": 0.88,
        "M2_raw": 0.93,
        "M2_freeze": 0.10,
        "M2_full": 0.94,
        "M3_raw": 0.87,
        "M3_full": 0.52,
        "M4_raw": 0.01,
        "M4_full": 0.70,
    }
    if TEST_RESULTS_FILE.exists():
        try:
            with open(TEST_RESULTS_FILE, encoding="utf-8") as f:
                test_res = json.load(f)
                for k, v in test_res.items():
                    if "th" in v:
                        defaults[k] = float(v["th"])
        except Exception:
            pass
    return defaults


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


def evaluate_with_threshold(scores, labels, th: float) -> dict:
    preds = [1 if s >= th else 0 for s in scores]
    return compute_metrics(labels, preds)


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


def run_transformers(pairs, model_key, hf_id, mode, ckpt_path=None, fixed_th=0.5):
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

    m = evaluate_with_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(fixed_th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
        "total": len(pairs), "pos_total": sum(labels), "neg_total": len(labels) - sum(labels)
    }
    return res


def run_astnn(pairs, model_key, mode, ckpt_path=None, fixed_th=0.5):
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

    m = evaluate_with_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(fixed_th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
        "total": len(pairs), "pos_total": sum(labels), "neg_total": len(labels) - sum(labels)
    }
    return res


def run_cpg(pairs, model_key, mode, ckpt_path=None, fixed_th=0.5):
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

    m = evaluate_with_threshold(scores, labels, fixed_th)
    res = {
        "th": round(float(fixed_th), 4),
        "tp": m["tp"], "tn": m["tn"], "fp": m["fp"], "fn": m["fn"],
        "f1": round(m["f1"] * 100, 2), "prec": round(m["prec"] * 100, 2),
        "rec": round(m["rec"] * 100, 2), "acc": round(m["acc"] * 100, 2),
        "total": len(pairs), "pos_total": sum(labels), "neg_total": len(labels) - sum(labels)
    }
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all")
    args = parser.parse_args()

    pairs = load_500_train_pairs()
    print(f"Loaded {len(pairs)} train pairs (balanced).")
    thresholds = load_test_thresholds()

    models_dispatch = {
        "M1_raw": lambda: run_transformers(pairs, "M1_raw", "microsoft/codebert-base", "raw", fixed_th=thresholds["M1_raw"]),
        "M1_freeze": lambda: run_transformers(pairs, "M1_freeze", "microsoft/codebert-base", "freeze", ckpt_path="results/M1_CodeBERT_Freeze/best_head.pt", fixed_th=thresholds["M1_freeze"]),
        "M1_full": lambda: run_transformers(pairs, "M1_full", "microsoft/codebert-base", "full", ckpt_path="results/exp_bcb_002/best_checkpoint.pt", fixed_th=thresholds["M1_full"]),
        "M2_raw": lambda: run_transformers(pairs, "M2_raw", "microsoft/graphcodebert-base", "raw", fixed_th=thresholds["M2_raw"]),
        "M2_freeze": lambda: run_transformers(pairs, "M2_freeze", "microsoft/graphcodebert-base", "freeze", ckpt_path="results/M2_GraphCodeBERT_Freeze/best_head.pt", fixed_th=thresholds["M2_freeze"]),
        "M2_full": lambda: run_transformers(pairs, "M2_full", "microsoft/graphcodebert-base", "full", ckpt_path="results/exp_bcb_003/best_checkpoint.pt", fixed_th=thresholds["M2_full"]),
        "M3_raw": lambda: run_astnn(pairs, "M3_raw", "raw", fixed_th=thresholds["M3_raw"]),
        "M3_full": lambda: run_astnn(pairs, "M3_full", "full", ckpt_path="results/exp_bcb_004/best_checkpoint.pt", fixed_th=thresholds["M3_full"]),
        "M4_raw": lambda: run_cpg(pairs, "M4_raw", "raw", fixed_th=thresholds["M4_raw"]),
        "M4_full": lambda: run_cpg(pairs, "M4_full", "full", ckpt_path="results/exp_bcb_005/best_checkpoint.pt", fixed_th=thresholds["M4_full"]),
    }

    results = {}
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    to_run = list(models_dispatch.keys()) if args.model == "all" else [args.model]
    for m in to_run:
        print(f"--- Đang đánh giá {m} trên 500 mẫu train (ngưỡng test = {thresholds.get(m)}) ---")
        res = models_dispatch[m]()
        results[m] = res
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # In bảng tổng hợp
    print(f"\n{'Mô hình':<20} | {'Ngưỡng':<7} | {'Đúng Clone':<12} | {'Đúng Non-clone':<15} | {'F1':<7} | {'Prec':<7} | {'Rec':<7} | {'Acc':<7}")
    print("-" * 92)
    for k in models_dispatch.keys():
        if k in results:
            v = results[k]
            print(f"{k:<20} | {v['th']:<7} | {v['tp']:>3}/250      | {v['tn']:>3}/250         | {v['f1']:<6.2f}% | {v['prec']:<6.2f}% | {v['rec']:<6.2f}% | {v['acc']:<6.2f}%")


if __name__ == "__main__":
    main()
