#!/usr/bin/env python3
"""Chạy lại kiểm thử toàn diện trên tập Test 5.000 cặp chuẩn 1:1 (2.500 clone / 2.500 non-clone)."""

import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAIRS_CACHE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
OUT_FILE = ROOT / "results" / "retested_5k_balanced_results.json"


def load_pairs() -> tuple[list[int], list[str], list[str]]:
    pairs = []
    with open(PAIRS_CACHE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    labels = [p["label"] for p in pairs]
    c1 = [p["func1"] for p in pairs]
    c2 = [p["func2"] for p in pairs]
    print(f"Loaded {len(pairs)} test pairs (Pos: {sum(labels)}, Neg: {len(labels) - sum(labels)})")
    return labels, c1, c2


def eval_transformer(hf_id: str, ckpt_path: str, max_length: int, labels: list[int], c1: list[str], c2: list[str], fixed_th: float = 0.5):
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
            scores.extend(torch.sigmoid(logits).cpu().tolist())

    y_true = np.array(labels)
    scores = np.array(scores)

    def calc(th):
        yp = (scores >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (yp == 1)))
        fp = int(np.sum((y_true == 0) & (yp == 1)))
        tn = int(np.sum((y_true == 0) & (yp == 0)))
        fn = int(np.sum((y_true == 1) & (yp == 0)))
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = 0.5
        return {
            "th": round(float(th), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "f1": round(f1 * 100, 2), "acc": round(acc * 100, 2),
            "prec": round(prec * 100, 2), "rec": round(rec * 100, 2),
            "roc_auc": round(auc * 100, 2),
        }

    fixed_res = calc(fixed_th)
    best_th, best_f1, best_res = 0.5, -1, None
    for th in np.linspace(0.01, 0.99, 99):
        res = calc(th)
        if res["f1"] > best_f1:
            best_f1, best_th, best_res = res["f1"], th, res

    return {"fixed": fixed_res, "optimal": best_res}


def eval_cpg(ckpt_path: str, labels: list[int], c1: list[str], c2: list[str], fixed_th: float = 0.5):
    from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device as cpg_to_device
    from torch_geometric.data import Batch, Data
    from src.representations.ast.extract import extract_tree

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

    def calc(th):
        yp = (scores >= th).astype(int)
        tp = int(np.sum((y_true == 1) & (yp == 1)))
        fp = int(np.sum((y_true == 0) & (yp == 1)))
        tn = int(np.sum((y_true == 0) & (yp == 0)))
        fn = int(np.sum((y_true == 1) & (yp == 0)))
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(y_true, scores))
        except Exception:
            auc = 0.5
        return {
            "th": round(float(th), 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "f1": round(f1 * 100, 2), "acc": round(acc * 100, 2),
            "prec": round(prec * 100, 2), "rec": round(rec * 100, 2),
            "roc_auc": round(auc * 100, 2),
        }

    fixed_res = calc(fixed_th)
    best_th, best_f1, best_res = 0.5, -1, None
    for th in np.linspace(0.01, 0.99, 99):
        res = calc(th)
        if res["f1"] > best_f1:
            best_f1, best_th, best_res = res["f1"], th, res

    return {"fixed": fixed_res, "optimal": best_res}


def main():
    labels, c1, c2 = load_pairs()
    all_results = {}

    # Load existing results from bcb_test_5k_results.json
    prev_file = ROOT / "results" / "bcb_test_5k_results.json"
    if prev_file.exists():
        with open(prev_file) as f:
            all_results = json.load(f)

    # 1. M1 384
    ckpt_m1_384 = ROOT / "results" / "exp_bcb_002_384" / "best_checkpoint.pt"
    if ckpt_m1_384.exists():
        print("\n>>> Đang đánh giá M1 384...")
        all_results["M1_384"] = eval_transformer("microsoft/codebert-base", str(ckpt_m1_384), 384, labels, c1, c2, fixed_th=0.88)
        print("M1 384 (Optimal):", all_results["M1_384"]["optimal"])

    # 2. M1 512
    ckpt_m1_512 = ROOT / "results" / "exp_bcb_002_512" / "best_checkpoint.pt"
    if ckpt_m1_512.exists():
        print("\n>>> Đang đánh giá M1 512...")
        all_results["M1_512_v1"] = eval_transformer("microsoft/codebert-base", str(ckpt_m1_512), 512, labels, c1, c2, fixed_th=0.90)
        print("M1 512 v1 (Optimal):", all_results["M1_512_v1"]["optimal"])

    # 3. M2 384
    ckpt_m2_384 = ROOT / "results" / "exp_bcb_003_384" / "best_checkpoint.pt"
    if ckpt_m2_384.exists():
        print("\n>>> Đang đánh giá M2 384...")
        all_results["M2_384"] = eval_transformer("microsoft/graphcodebert-base", str(ckpt_m2_384), 384, labels, c1, c2, fixed_th=0.94)
        print("M2 384 (Optimal):", all_results["M2_384"]["optimal"])

    # 4. M2 512
    ckpt_m2_512 = ROOT / "results" / "exp_bcb_003_512" / "best_checkpoint.pt"
    if ckpt_m2_512.exists():
        print("\n>>> Đang đánh giá M2 512...")
        all_results["M2_512_v1"] = eval_transformer("microsoft/graphcodebert-base", str(ckpt_m2_512), 512, labels, c1, c2, fixed_th=0.64)
        print("M2 512 v1 (Optimal):", all_results["M2_512_v1"]["optimal"])

    # 5. M4 Opt
    ckpt_m4_opt = ROOT / "results" / "exp_bcb_005_opt" / "best_checkpoint.pt"
    if ckpt_m4_opt.exists():
        print("\n>>> Đang đánh giá M4 Opt...")
        all_results["M4_opt"] = eval_cpg(str(ckpt_m4_opt), labels, c1, c2, fixed_th=0.62)
        print("M4 Opt (Optimal):", all_results["M4_opt"]["optimal"])

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nĐã lưu toàn bộ kết quả kiểm thử vào {OUT_FILE}!")


if __name__ == "__main__":
    main()
