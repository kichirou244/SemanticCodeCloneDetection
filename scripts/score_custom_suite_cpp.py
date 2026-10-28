"""Chấm điểm và phân tích định tính toàn diện bộ kiểm thử C/C++ Custom Suite (720 cặp, tỉ lệ 1:5).

Thực hiện đánh giá trên 720 cặp mã nguồn C/C++ (120 clone, 600 non-clone) qua 4 mô hình tinh chỉnh trên POJ-104:
- M1 (CodeBERT - Sequence Transformer) [Ngưỡng: 0.21]
- M2 (GraphCodeBERT - Sequence + DFG) [Ngưỡng: 0.66]
- M3 (ASTNN - Tree RvNN) [Ngưỡng: 0.42]
- M4 (CPG-GNN - Joern CPG / GAT) [Ngưỡng: 0.47]

Xuất kết quả ra `workspace/custom_suite_cpp_scores.json`.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "custom_suite"
OUT_PATH = ROOT / "workspace" / "custom_suite_cpp_scores.json"

THRESHOLDS = {
    "M1": 0.21,
    "M2": 0.66,
    "M3": 0.42,
    "M4": 0.47,
}


def load_pairs() -> list[dict]:
    manifest = DATA_DIR / "pairs.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Không tìm thấy file {manifest}")
    pairs = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    codes = {p.name: p.read_text(encoding="utf-8") for p in DATA_DIR.glob("*.cpp")}
    for r in pairs:
        r["code_a"] = codes[r["a"]]
        r["code_b"] = codes[r["b"]]
    return pairs


def score_m3_astnn(pairs: list[dict], device: torch.device) -> dict[str, float]:
    """Chấm điểm bằng mô hình cây M3 (ASTNN) trên C/C++."""
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees

    ckpt_path = ROOT / "results" / "exp004_astnn_poj104" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"Cảnh báo: Không tìm thấy checkpoint M3 tại {ckpt_path}")
        return {}

    print(">>> Nạp trọng số M3 (ASTNN) C/C++...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=27)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    compiled_cache: dict[str, Any] = {}
    for p in DATA_DIR.glob("*.cpp"):
        try:
            compiled_cache[p.name] = compile_tree(extract_tree(p.read_text(encoding="utf-8"), language="cpp"))
        except Exception:
            pass

    scores = {}
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            batch_a_compiled = [compiled_cache[p["a"]] for p in chunk if p["a"] in compiled_cache and p["b"] in compiled_cache]
            batch_b_compiled = [compiled_cache[p["b"]] for p in chunk if p["a"] in compiled_cache and p["b"] in compiled_cache]
            valid_chunk = [p for p in chunk if p["a"] in compiled_cache and p["b"] in compiled_cache]

            if not valid_chunk:
                continue

            batch_a = astnn_to_device(collate_trees(batch_a_compiled), device)
            batch_b = astnn_to_device(collate_trees(batch_b_compiled), device)

            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits).tolist()
            if isinstance(probs, float):
                probs = [probs]
            for p, prob in zip(valid_chunk, probs):
                scores[p["pair_id"]] = float(prob)

    return scores


def score_m4_cpg(pairs: list[dict], device: torch.device) -> dict[str, float]:
    """Chấm điểm bằng mô hình đồ thị M4 (CPG-GNN) trên C/C++."""
    from src.models.cpg_gnn.model import CPGGNNPairClassifier
    from torch_geometric.data import Batch, Data
    from src.representations.ast.extract import extract_tree

    ckpt_path = ROOT / "results" / "exp005_cpg_gnn_poj104" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"Cảnh báo: Không tìm thấy checkpoint M4 tại {ckpt_path}")
        return {}

    print(">>> Nạp trọng số M4 (CPG-GNN) C/C++...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    def code_to_graph_data(code: str) -> Data:
        tree = extract_tree(code, language="cpp")
        labels = tree["label"]
        parents = tree["parent"]
        n = len(labels)
        src, dst = [], []
        for i in range(1, n):
            p = int(parents[i])
            if 0 <= p < n and p != i:
                src.extend([p, i])
                dst.extend([i, p])
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_type = torch.zeros(len(src), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)

        node_type = torch.tensor(labels % 13, dtype=torch.long)
        x = torch.ones((n, 1), dtype=torch.float32)
        return Data(node_type=node_type, x=x, edge_index=edge_index, edge_type=edge_type)

    graph_cache: dict[str, Data] = {}
    for p in DATA_DIR.glob("*.cpp"):
        try:
            graph_cache[p.name] = code_to_graph_data(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    scores = {}
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            valid_chunk = [p for p in chunk if p["a"] in graph_cache and p["b"] in graph_cache]
            if not valid_chunk:
                continue

            g_a_list = [graph_cache[p["a"]] for p in valid_chunk]
            g_b_list = [graph_cache[p["b"]] for p in valid_chunk]

            ga = Batch.from_data_list(g_a_list).to(device)
            gb = Batch.from_data_list(g_b_list).to(device)

            logits = model(ga, gb)
            probs = torch.sigmoid(logits).tolist()
            if isinstance(probs, float):
                probs = [probs]
            for p, prob in zip(valid_chunk, probs):
                scores[p["pair_id"]] = float(prob)

    return scores


def score_transformer(pairs: list[dict], hf_id: str, model_id: str, device: torch.device) -> dict[str, float]:
    """Chấm điểm chuỗi mã nguồn qua Transformer (CodeBERT M1 / GraphCodeBERT M2) với checkpoint tinh chỉnh trên POJ-104."""
    from transformers import AutoTokenizer, AutoModel
    from src.models.transformer_pair import TransformerPairClassifier

    ckpt_path = ROOT / "results" / ("exp002_codebert_poj104_512" if model_id == "M1" else "exp003_graphcodebert_poj104_512") / "best_checkpoint.pt"
    print(f">>> Khởi tạo mô hình {model_id} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    encoder = AutoModel.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, encoder=encoder)

    if ckpt_path.exists():
        print(f"    Nạp trọng số tinh chỉnh từ {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print(f"    Cảnh báo: Không tìm thấy checkpoint tại {ckpt_path}, dùng pretrained gốc.")

    model.to(device)
    model.eval()

    scores = {}
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            codes_a = [p["code_a"] for p in chunk]
            codes_b = [p["code_b"] for p in chunk]

            enc_a = tokenizer(
                codes_a,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            enc_b = tokenizer(
                codes_b,
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            logit = model(enc_a, enc_b)
            probs = torch.sigmoid(logit).tolist()
            if isinstance(probs, float):
                probs = [probs]
            for p, prob in zip(chunk, probs):
                scores[p["pair_id"]] = float(prob)

    return scores


def evaluate_metrics(pairs: list[dict], scores: dict[str, float], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    by_category = defaultdict(lambda: {"total": 0, "correct": 0, "preds": []})

    for p in pairs:
        pid = p["pair_id"]
        if pid not in scores:
            continue
        prob = scores[pid]
        pred = 1 if prob >= threshold else 0
        actual = p["label"]
        cat = p["category"]

        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 0:
            tn += 1
        else:
            fn += 1

        is_correct = (pred == actual)
        by_category[cat]["total"] += 1
        if is_correct:
            by_category[cat]["correct"] += 1
        by_category[cat]["preds"].append({"pair_id": pid, "prob": prob, "pred": pred, "actual": actual})

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "threshold": threshold,
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "by_category": {
            cat: {
                "total": info["total"],
                "correct": info["correct"],
                "accuracy": info["correct"] / info["total"] if info["total"] > 0 else 0.0,
            }
            for cat, info in by_category.items()
        },
    }


def main():
    pairs = load_pairs()
    n_clones = sum(1 for p in pairs if p["label"] == 1)
    n_nonclones = sum(1 for p in pairs if p["label"] == 0)
    print(f"Tổng số cặp kiểm thử: {len(pairs)} ({n_clones} Clones, {n_nonclones} Non-clones, tỉ lệ 1:{n_nonclones//n_clones})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán: {device}")

    all_scores: dict[str, dict[str, float]] = {}

    # 1. Chấm M3 (ASTNN)
    try:
        all_scores["M3"] = score_m3_astnn(pairs, device)
    except Exception as e:
        print(f"Lỗi khi chấm M3: {e}")
        all_scores["M3"] = {}

    # 2. Chấm M4 (CPG-GNN)
    try:
        all_scores["M4"] = score_m4_cpg(pairs, device)
    except Exception as e:
        print(f"Lỗi khi chấm M4: {e}")
        all_scores["M4"] = {}

    # 3. Chấm M1 (CodeBERT)
    try:
        all_scores["M1"] = score_transformer(pairs, "microsoft/codebert-base", "M1", device)
    except Exception as e:
        print(f"Lỗi khi chấm M1: {e}")
        all_scores["M1"] = {}

    # 4. Chấm M2 (GraphCodeBERT)
    try:
        all_scores["M2"] = score_transformer(pairs, "microsoft/graphcodebert-base", "M2", device)
    except Exception as e:
        print(f"Lỗi khi chấm M2: {e}")
        all_scores["M2"] = {}

    # Tổng hợp ma trận nhầm lẫn và báo cáo
    model_metrics = {}
    categories_order = [
        "clone_type1",
        "clone_type2",
        "clone_type3",
        "clone_type4",
        "nonclone_trap",
        "nonclone_cross",
    ]

    cat_labels = {
        "clone_type1": "Type-1 (Rename)",
        "clone_type2": "Type-2 (Types/Wrap)",
        "clone_type3": "Type-3 (Control Flow)",
        "clone_type4": "Type-4 (Semantic)",
        "nonclone_trap": "Non-clone Khó (Traps)",
        "nonclone_kho": "Non-clone Khó (Traps)",
        "nonclone_cross": "Non-clone Cross",
    }

    print("\n" + "=" * 90)
    print(f"BẢNG TỔNG HỢP HIỆU NĂNG ĐỊNH TÍNH TRÊN CUSTOM SUITE C/C++ ({len(pairs)} CẶP - TỈ LỆ 1:5)")
    print("=" * 90)
    print(f"{'Model':<6} | {'Threshold':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10} | {'TP/FP/TN/FN'}")
    print("-" * 90)

    for mid in ["M1", "M2", "M3", "M4"]:
        if mid in all_scores and all_scores[mid]:
            m = evaluate_metrics(pairs, all_scores[mid], THRESHOLDS[mid])
            model_metrics[mid] = m
            tf_str = f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}"
            print(f"{mid:<6} | {m['threshold']:<10.2f} | {m['f1']*100:<9.2f}% | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}% | {m['accuracy']*100:<9.2f}% | {tf_str}")
        else:
            print(f"{mid:<6} | {'N/A':<10} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<10} | -")

    print("\n" + "=" * 90)
    print("CHI TIẾT ĐỘ CHÍNH XÁC THEO TỪNG LOẠI BIẾN THỂ VÀ BẪY NON-CLONE (%)")
    print("=" * 90)
    header = f"{'Loại kiểm thử':<26} | {'Số cặp':<7} | " + " | ".join(f"{mid:<8}" for mid in ["M1", "M2", "M3", "M4"])
    print(header)
    print("-" * 90)

    for cat in categories_order:
        row = [f"{cat_labels[cat]:<26}"]
        n_pairs = sum(1 for p in pairs if p["category"] == cat)
        row.append(f"{n_pairs:<7}")
        for mid in ["M1", "M2", "M3", "M4"]:
            if mid in model_metrics and cat in model_metrics[mid]["by_category"]:
                acc_cat = model_metrics[mid]["by_category"][cat]["accuracy"] * 100
                row.append(f"{acc_cat:<7.1f}%")
            else:
                row.append(f"{'-':<8}")
        print(" | ".join(row))
    print("=" * 90)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export_data = {
        "summary": model_metrics,
        "pairs_detail": [
            {
                "pair_id": p["pair_id"],
                "a": p["a"],
                "b": p["b"],
                "category": p["category"],
                "label": p["label"],
                "scores": {mid: all_scores[mid].get(p["pair_id"]) for mid in ["M1", "M2", "M3", "M4"] if mid in all_scores},
            }
            for p in pairs
        ],
    }
    OUT_PATH.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu toàn bộ kết quả chi tiết vào: {OUT_PATH}")


if __name__ == "__main__":
    main()
