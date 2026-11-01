#!/usr/bin/env python3
"""Danh gia 4 mo hinh (10 bien the) tren tap benchmark CODER HUSC (C/C++)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from transformers import AutoModel, AutoTokenizer

from scripts.fast_native_cpg import FastNativeCPGExtractor
from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
from src.models.cpg_gnn.model import CPGGNNPairClassifier
from src.models.transformer_pair import TransformerPairClassifier, tokenize_side
from src.representations.ast.extract import collate_trees, compile_tree, extract_tree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("evaluate_coder")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
BATCH_SIZE = 32


def load_coder_dataset(path: str = "data/processed/coder_benchmark/coder_test_pairs.jsonl") -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
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

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1 * 100), 2),
        "precision": round(float(prec * 100), 2),
        "recall": round(float(rec * 100), 2),
        "accuracy": round(float(acc * 100), 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "correct_clone": f"{tp} / {tp + fn}",
        "correct_non_clone": f"{tn} / {tn + fp}",
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    best_th, best_f1, best_m = 0.5, -1.0, {}
    for th in np.linspace(0.01, 0.99, 99):
        m = compute_metrics(y_true, y_prob, threshold=th)
        if m["f1"] > best_f1:
            best_f1, best_th, best_m = m["f1"], th, m
    return best_th, best_m


def run_zero_shot(hf_id: str, pairs: list[dict]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    encoder = AutoModel.from_pretrained(hf_id).to(DEVICE)
    encoder.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            inp1, _ = tokenize_side(tokenizer, [p["func1"] for p in batch], max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, [p["func2"] for p in batch], max_length=MAX_LEN)
            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}
            h1 = encoder(**inp1).last_hidden_state[:, 0, :]
            h2 = encoder(**inp2).last_hidden_state[:, 0, :]
            cos = F.cosine_similarity(h1, h2, dim=-1)
            probs.extend(((cos + 1.0) / 2.0).clamp(0.0, 1.0).cpu().numpy())
    return np.array(probs)


def run_freeze_model(hf_id: str, head_ckpt_path: str, pairs: list[dict]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, freeze_backbone=True).to(DEVICE)
    ckpt = torch.load(head_ckpt_path, map_location=DEVICE)
    model.head.load_state_dict(ckpt.get("head_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            inp1, _ = tokenize_side(tokenizer, [p["func1"] for p in batch], max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, [p["func2"] for p in batch], max_length=MAX_LEN)
            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}
            probs.extend(torch.sigmoid(model(inp1, inp2)).cpu().numpy())
    return np.array(probs)


def run_full_finetuned_model(hf_id: str, full_ckpt_path: str, pairs: list[dict]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, freeze_backbone=False).to(DEVICE)
    ckpt = torch.load(full_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            inp1, _ = tokenize_side(tokenizer, [p["func1"] for p in batch], max_length=MAX_LEN)
            inp2, _ = tokenize_side(tokenizer, [p["func2"] for p in batch], max_length=MAX_LEN)
            inp1 = {k: v.to(DEVICE) for k, v in inp1.items()}
            inp2 = {k: v.to(DEVICE) for k, v in inp2.items()}
            probs.extend(torch.sigmoid(model(inp1, inp2)).cpu().numpy())
    return np.array(probs)


def run_astnn_model(ckpt_path: str | None, pairs: list[dict], ast_cache: dict) -> np.ndarray:
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            ba = astnn_to_device(collate_trees([ast_cache[p["sub1"]] for p in batch]), DEVICE)
            bb = astnn_to_device(collate_trees([ast_cache[p["sub2"]] for p in batch]), DEVICE)
            probs.extend(torch.sigmoid(model(ba, bb)).cpu().numpy())
    return np.array(probs)


def run_cpg_model(ckpt_path: str | None, pairs: list[dict], cpg_cache: dict) -> np.ndarray:
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2).to(DEVICE)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            ga = Batch.from_data_list([cpg_cache[p["sub1"]] for p in batch]).to(DEVICE)
            gb = Batch.from_data_list([cpg_cache[p["sub2"]] for p in batch]).to(DEVICE)
            probs.extend(torch.sigmoid(model(ga, gb)).cpu().numpy())
    return np.array(probs)


def evaluate_breakdown(pairs: list[dict], probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    counts, correct = {}, {}
    for p, pred in zip(pairs, preds):
        ctype = p["clone_type"]
        counts[ctype] = counts.get(ctype, 0) + 1
        if pred == p["label"]:
            correct[ctype] = correct.get(ctype, 0) + 1

    return {
        ctype: {"correct": correct.get(ctype, 0), "total": total, "ratio": f"{correct.get(ctype, 0)} / {total}"}
        for ctype, total in counts.items()
    }


def main():
    pairs = load_coder_dataset()
    y_true = np.array([p["label"] for p in pairs])

    unique_codes = {p["sub1"]: p["func1"] for p in pairs}
    unique_codes.update({p["sub2"]: p["func2"] for p in pairs})

    ast_cache = {sid: compile_tree(extract_tree(code, language="cpp")) for sid, code in unique_codes.items()}
    cpg_ext = FastNativeCPGExtractor("cpp")
    cpg_cache = {sid: cpg_ext.parse_to_cpg(code) for sid, code in unique_codes.items()}

    experiments = [
        ("M1 Chưa huấn luyện", lambda: run_zero_shot("microsoft/codebert-base", pairs)),
        ("M1 Đóng băng 12 tầng", lambda: run_freeze_model("microsoft/codebert-base", "results/M1_CodeBERT_Freeze/best_head.pt", pairs)),
        ("M1 Fine-tuning toàn bộ", lambda: run_full_finetuned_model("microsoft/codebert-base", "results/exp_bcb_002/best_checkpoint.pt", pairs)),
        ("M2 Chưa huấn luyện", lambda: run_zero_shot("microsoft/graphcodebert-base", pairs)),
        ("M2 Đóng băng 12 tầng", lambda: run_freeze_model("microsoft/graphcodebert-base", "results/M2_GraphCodeBERT_Freeze/best_head.pt", pairs)),
        ("M2 Fine-tuning toàn bộ", lambda: run_full_finetuned_model("microsoft/graphcodebert-base", "results/exp_bcb_003/best_checkpoint.pt", pairs)),
        ("M3 Chưa huấn luyện", lambda: run_astnn_model(None, pairs, ast_cache)),
        ("M3 Fine-tuning toàn bộ", lambda: run_astnn_model("results/exp_bcb_004/best_checkpoint.pt", pairs, ast_cache)),
        ("M4 Chưa huấn luyện", lambda: run_cpg_model(None, pairs, cpg_cache)),
        ("M4 Fine-tuning toàn bộ", lambda: run_cpg_model("results/exp_bcb_005/best_checkpoint.pt", pairs, cpg_cache)),
    ]

    all_results = []
    for name, runner in experiments:
        logger.info("Danh gia: %s", name)
        probs = runner()
        best_th, m = find_best_threshold(y_true, probs)
        breakdown = evaluate_breakdown(pairs, probs, best_th)
        all_results.append({
            "model": name,
            "threshold": m["threshold"],
            "correct_clone": m["correct_clone"],
            "correct_non_clone": m["correct_non_clone"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "accuracy": m["accuracy"],
            "breakdown": breakdown,
        })

    out_file = Path("results/coder_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Da ghi ket qua vao %s", out_file)


if __name__ == "__main__":
    main()
