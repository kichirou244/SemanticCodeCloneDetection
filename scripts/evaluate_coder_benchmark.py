#!/usr/bin/env python3
"""Chay kiem thu toan bo cac phuong phap (M1, M2, M3, M4) tren tap CODER HUSC (C/C++).

Cac mo hinh danh gia:
1. M1 Chưa huấn luyện (Zero-shot Cosine tu microsoft/codebert-base)
2. M1 Đóng băng 12 tầng (Backbone Frozen + Trained Head tu BigCloneBench)
3. M1 Fine-tuning toàn bộ (Full Fine-tuning tu exp_bcb_002)
4. M2 Chưa huấn luyện (Zero-shot Cosine tu microsoft/graphcodebert-base)
5. M2 Đóng băng 12 tầng (Backbone Frozen + Trained Head tu BigCloneBench)
6. M2 Fine-tuning toàn bộ (Full Fine-tuning tu exp_bcb_003)
7. M3 Chưa huấn luyện (ASTNN Untrained)
8. M3 Fine-tuning toàn bộ (ASTNN tu exp_bcb_004)
9. M4 Chưa huấn luyện (CPG-GNN Untrained)
10. M4 Fine-tuning toàn bộ (CPG-GNN tu exp_bcb_005)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Them root vao sys.path de import duoc src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from transformers import AutoModel, AutoTokenizer

from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from src.models.transformer_pair import TransformerPairClassifier, tokenize_side
from src.representations.ast.extract import collate_trees, compile_tree, extract_tree
from scripts.fast_native_cpg import FastNativeCPGExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_coder")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 32


def load_coder_dataset(path: str = "data/processed/coder_benchmark/coder_test_pairs.jsonl") -> list[dict]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    pos_total = tp + fn
    neg_total = tn + fp
    total = len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1 * 100), 2),
        "precision": round(float(precision * 100), 2),
        "recall": round(float(recall * 100), 2),
        "accuracy": round(float(acc * 100), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_clone": f"{tp} / {pos_total}",
        "correct_non_clone": f"{tn} / {neg_total}",
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_th = 0.5
    best_f1 = -1.0
    best_metrics = {}
    for th in np.linspace(0.01, 0.99, 99):
        m = compute_metrics(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m
    return best_th, best_metrics


def run_zero_shot(hf_id: str, pairs: list[dict]) -> np.ndarray:
    """Tinh Cosine Similarity tren vector [CLS] cua checkpoint pre-trained goc."""
    logger.info(f"--- Chay Zero-shot Cosine: {hf_id} ---")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    encoder = AutoModel.from_pretrained(hf_id).to(DEVICE)
    encoder.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            funcs1 = [p["func1"] for p in batch_pairs]
            funcs2 = [p["func2"] for p in batch_pairs]

            inp1, _ = tokenize_side(tokenizer, funcs1, max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, funcs2, max_length=MAX_LEN)

            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}

            h1 = encoder(**inp1).last_hidden_state[:, 0, :]
            h2 = encoder(**inp2).last_hidden_state[:, 0, :]

            cos_sim = F.cosine_similarity(h1, h2, dim=-1)
            prob = ((cos_sim + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy()
            probs.extend(prob)

    return np.array(probs)


def run_freeze_model(hf_id: str, head_ckpt_path: str, pairs: list[dict]) -> np.ndarray:
    """Chay model Dong bang 12 tang (Backbone Frozen + Trained Head)."""
    logger.info(f"--- Chay Dong bang 12 tang: {hf_id} (Head: {head_ckpt_path}) ---")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, freeze_backbone=True).to(DEVICE)

    ckpt = torch.load(head_ckpt_path, map_location=DEVICE)
    if "head_state" in ckpt:
        model.head.load_state_dict(ckpt["head_state"])
    else:
        model.head.load_state_dict(ckpt)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            funcs1 = [p["func1"] for p in batch_pairs]
            funcs2 = [p["func2"] for p in batch_pairs]

            inp1, _ = tokenize_side(tokenizer, funcs1, max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, funcs2, max_length=MAX_LEN)

            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}

            logits = model(inp1, inp2)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob)

    return np.array(probs)


def run_full_finetuned_model(hf_id: str, full_ckpt_path: str, pairs: list[dict]) -> np.ndarray:
    """Chay model Fine-tuning toan bo."""
    logger.info(f"--- Chay Fine-tuning toan bo: {hf_id} (Checkpoint: {full_ckpt_path}) ---")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, freeze_backbone=False).to(DEVICE)

    ckpt = torch.load(full_ckpt_path, map_location=DEVICE)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            funcs1 = [p["func1"] for p in batch_pairs]
            funcs2 = [p["func2"] for p in batch_pairs]

            inp1, _ = tokenize_side(tokenizer, funcs1, max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, funcs2, max_length=MAX_LEN)

            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}

            logits = model(inp1, inp2)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob)

    return np.array(probs)


def run_astnn_model(ckpt_path: str | None, pairs: list[dict], ast_cache: dict) -> np.ndarray:
    """Chay model M3 (ASTNN)."""
    name = f"ASTNN (Checkpoint: {ckpt_path})" if ckpt_path else "ASTNN Chưa huấn luyện"
    logger.info(f"--- Chay {name} ---")
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            trees_a = [ast_cache[p["sub1"]] for p in batch_pairs]
            trees_b = [ast_cache[p["sub2"]] for p in batch_pairs]

            ba = astnn_to_device(collate_trees(trees_a), DEVICE)
            bb = astnn_to_device(collate_trees(trees_b), DEVICE)

            logits = model(ba, bb)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob)

    return np.array(probs)


def run_cpg_model(ckpt_path: str | None, pairs: list[dict], cpg_cache: dict) -> np.ndarray:
    """Chay model M4 (CPG-GNN)."""
    name = f"CPG-GNN (Checkpoint: {ckpt_path})" if ckpt_path else "CPG-GNN Chưa huấn luyện"
    logger.info(f"--- Chay {name} ---")
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2).to(DEVICE)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch_pairs = pairs[i : i + BATCH_SIZE]
            cpgs_a = [cpg_cache[p["sub1"]] for p in batch_pairs]
            cpgs_b = [cpg_cache[p["sub2"]] for p in batch_pairs]

            ga = Batch.from_data_list(cpgs_a).to(DEVICE)
            gb = Batch.from_data_list(cpgs_b).to(DEVICE)

            logits = model(ga, gb)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.extend(prob)

    return np.array(probs)


def evaluate_breakdown(pairs: list[dict], probs: np.ndarray, threshold: float) -> dict:
    """Phan tich ket qua du doan theo tung phan tang clone type."""
    preds = (probs >= threshold).astype(int)
    breakdown = {}

    type_counts = {}
    type_correct = {}

    for p, pred in zip(pairs, preds):
        ctype = p["clone_type"]
        lbl = p["label"]
        type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        is_correct = (pred == lbl)
        if is_correct:
            type_correct[ctype] = type_correct.get(ctype, 0) + 1

    for ctype, total in type_counts.items():
        correct = type_correct.get(ctype, 0)
        breakdown[ctype] = {
            "correct": correct,
            "total": total,
            "ratio": f"{correct} / {total}",
        }

    return breakdown


def main():
    pairs = load_coder_dataset()
    y_true = np.array([p["label"] for p in pairs])
    logger.info(f"Nạp thanh cong {len(pairs)} cap kiem thu CODER (Positive: {sum(y_true)}, Negative: {len(y_true) - sum(y_true)})")

    # Pre-extract AST va CPG cho tat ca ma nguon duy nhat
    unique_codes = {}
    for p in pairs:
        unique_codes[p["sub1"]] = p["func1"]
        unique_codes[p["sub2"]] = p["func2"]

    logger.info(f"Trich xuat AST cho {len(unique_codes)} doan ma...")
    ast_cache = {}
    for sid, code in unique_codes.items():
        ast_cache[sid] = compile_tree(extract_tree(code, language="cpp"))

    logger.info(f"Trich xuat CPG cho {len(unique_codes)} doan ma...")
    cpg_ext = FastNativeCPGExtractor("cpp")
    cpg_cache = {}
    for sid, code in unique_codes.items():
        cpg_cache[sid] = cpg_ext.parse_to_cpg(code)

    experiments = [
        {
            "name": "M1 Chưa huấn luyện",
            "runner": lambda: run_zero_shot("microsoft/codebert-base", pairs),
        },
        {
            "name": "M1 Đóng băng 12 tầng",
            "runner": lambda: run_freeze_model("microsoft/codebert-base", "results/M1_CodeBERT_Freeze/best_head.pt", pairs),
        },
        {
            "name": "M1 Fine-tuning toàn bộ",
            "runner": lambda: run_full_finetuned_model("microsoft/codebert-base", "results/exp_bcb_002/best_checkpoint.pt", pairs),
        },
        {
            "name": "M2 Chưa huấn luyện",
            "runner": lambda: run_zero_shot("microsoft/graphcodebert-base", pairs),
        },
        {
            "name": "M2 Đóng băng 12 tầng",
            "runner": lambda: run_freeze_model("microsoft/graphcodebert-base", "results/M2_GraphCodeBERT_Freeze/best_head.pt", pairs),
        },
        {
            "name": "M2 Fine-tuning toàn bộ",
            "runner": lambda: run_full_finetuned_model("microsoft/graphcodebert-base", "results/exp_bcb_003/best_checkpoint.pt", pairs),
        },
        {
            "name": "M3 Chưa huấn luyện",
            "runner": lambda: run_astnn_model(None, pairs, ast_cache),
        },
        {
            "name": "M3 Fine-tuning toàn bộ",
            "runner": lambda: run_astnn_model("results/exp_bcb_004/best_checkpoint.pt", pairs, ast_cache),
        },
        {
            "name": "M4 Chưa huấn luyện",
            "runner": lambda: run_cpg_model(None, pairs, cpg_cache),
        },
        {
            "name": "M4 Fine-tuning toàn bộ",
            "runner": lambda: run_cpg_model("results/exp_bcb_005/best_checkpoint.pt", pairs, cpg_cache),
        },
    ]

    all_results = []

    for exp in experiments:
        logger.info(f"\n========================================\nDang danh gia: {exp['name']}\n========================================")
        probs = exp["runner"]()
        best_th, metrics = find_best_threshold(y_true, probs)
        breakdown = evaluate_breakdown(pairs, probs, best_th)

        result_item = {
            "model": exp["name"],
            "threshold": metrics["threshold"],
            "correct_clone": metrics["correct_clone"],
            "correct_non_clone": metrics["correct_non_clone"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "accuracy": metrics["accuracy"],
            "breakdown": breakdown,
        }
        all_results.append(result_item)
        logger.info(f"Ket qua {exp['name']}:")
        logger.info(f"  F1: {metrics['f1']}%, Acc: {metrics['accuracy']}%, Correct Clone: {metrics['correct_clone']}, Correct Non-clone: {metrics['correct_non_clone']}")
        logger.info(f"  Phan ra chi tiet: {breakdown}")

    out_file = Path("results/coder_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nDa luu toan bo ket qua vao: {out_file}")


if __name__ == "__main__":
    main()
