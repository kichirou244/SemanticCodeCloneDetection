"""
Script: score_cpp_with_java_models.py
Mục tiêu: Đánh giá khả năng chuyển dịch ngôn ngữ (Cross-lingual Transfer):
Dùng toàn bộ 4 mô hình (M1, M2, M3, M4) được HUẤN LUYỆN TRÊN JAVA (BigCloneBench)
để chấm điểm trực tiếp trên tập kiểm thử C/C++ Custom Suite (720 cặp, tỉ lệ 1:5).
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = ROOT / "data" / "custom_suite"
OUT_PATH = ROOT / "workspace" / "custom_suite_cpp_with_java_models_scores.json"

# Nạp dữ liệu C/C++
manifest = DATA_DIR / "pairs.jsonl"
pairs = [json.loads(ln) for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
codes = {p.name: p.read_text(encoding="utf-8") for p in DATA_DIR.glob("*.cpp")}
for r in pairs:
    r["code_a"] = codes[r["a"]]
    r["code_b"] = codes[r["b"]]

labels = np.array([p["label"] for p in pairs])
print(f"Tổng số cặp C/C++: {len(pairs)} (Clone: {sum(labels)}, Non-clone: {len(labels)-sum(labels)})")

results_dict = {}

# --------------------------------------------------------------------------
# 1. M1 & M2 Zero-shot (Pretrained Cosine)
# --------------------------------------------------------------------------
def eval_zero_shot(hf_id: str, name: str):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf_id)
    enc = AutoModel.from_pretrained(hf_id).to(DEVICE).eval()
    
    sims = []
    codes_a = [p["code_a"] for p in pairs]
    codes_b = [p["code_b"] for p in pairs]
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            t1 = tok(codes_a[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            t2 = tok(codes_b[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            h1 = enc(**t1).last_hidden_state[:, 0, :]
            h2 = enc(**t2).last_hidden_state[:, 0, :]
            sim = torch.cosine_similarity(h1, h2, dim=-1).cpu().numpy()
            sims.extend(sim.tolist() if sim.ndim > 0 else [float(sim)])
    sims = np.array(sims)
    
    # Tìm ngưỡng tốt nhất
    best_f1, best_th, best_info = -1, 0.5, None
    for th in np.linspace(0.1, 0.999, 180):
        preds = (sims >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            best_info = {
                'th': th, 'f1': f1, 'prec': precision_score(labels, preds, zero_division=0),
                'rec': recall_score(labels, preds, zero_division=0),
                'acc': accuracy_score(labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    auc = roc_auc_score(labels, sims)
    print(f"[{name} Zero-shot] ROC-AUC: {auc*100:.2f}% | th={best_info['th']:.4f} | F1={best_info['f1']*100:.2f}% | Prec={best_info['prec']*100:.2f}% | Rec={best_info['rec']*100:.2f}% | Acc={best_info['acc']*100:.2f}% | {best_info['tp']} / {best_info['fp']} / {best_info['tn']} / {best_info['fn']}")
    return best_info

# --------------------------------------------------------------------------
# 2. M1 & M2 Fine-tuned trên Java (BigCloneBench)
# --------------------------------------------------------------------------
def eval_transformer_bcb(ckpt_path: Path, hf_id: str, name: str, frozen_th: float):
    from transformers import AutoTokenizer, AutoModel
    from src.models.transformer_pair import TransformerPairClassifier
    
    tok = AutoTokenizer.from_pretrained(hf_id)
    enc = AutoModel.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, encoder=enc)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    
    probs = []
    codes_a = [p["code_a"] for p in pairs]
    codes_b = [p["code_b"] for p in pairs]
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            t1 = tok(codes_a[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            t2 = tok(codes_b[i:i+batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            logits = model(t1, t2)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
    probs = np.array(probs)
    
    # Đánh giá tại ngưỡng đóng băng
    preds_frozen = (probs >= frozen_th).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(labels, preds_frozen).ravel()
    frozen_info = {
        'th': frozen_th, 'f1': f1_score(labels, preds_frozen, zero_division=0),
        'prec': precision_score(labels, preds_frozen, zero_division=0),
        'rec': recall_score(labels, preds_frozen, zero_division=0),
        'acc': accuracy_score(labels, preds_frozen),
        'tp': int(tp_f), 'fp': int(fp_f), 'tn': int(tn_f), 'fn': int(fn_f)
    }
    
    # Tìm ngưỡng tối ưu trên tập test
    best_f1, best_th, best_info = -1, 0.5, None
    for th in np.linspace(0.01, 0.99, 100):
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            best_info = {
                'th': th, 'f1': f1, 'prec': precision_score(labels, preds, zero_division=0),
                'rec': recall_score(labels, preds, zero_division=0),
                'acc': accuracy_score(labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    auc = roc_auc_score(labels, probs)
    print(f"[{name} Fine-tuned BCB (Java)] ROC-AUC: {auc*100:.2f}%")
    print(f"  * Frozen th={frozen_info['th']:.2f}: F1={frozen_info['f1']*100:.2f}% | Prec={frozen_info['prec']*100:.2f}% | Rec={frozen_info['rec']*100:.2f}% | Acc={frozen_info['acc']*100:.2f}% | {frozen_info['tp']} / {frozen_info['fp']} / {frozen_info['tn']} / {frozen_info['fn']}")
    print(f"  * Best th={best_info['th']:.2f}: F1={best_info['f1']*100:.2f}% | Prec={best_info['prec']*100:.2f}% | Rec={best_info['rec']*100:.2f}% | Acc={best_info['acc']*100:.2f}% | {best_info['tp']} / {best_info['fp']} / {best_info['tn']} / {best_info['fn']}")
    return frozen_info, best_info, probs

# --------------------------------------------------------------------------
# 3. M3 ASTNN Fine-tuned trên Java (BigCloneBench)
# --------------------------------------------------------------------------
def eval_astnn_bcb(ckpt_path: Path, frozen_th: float = 0.52):
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    
    compiled_cache = {}
    for p in DATA_DIR.glob("*.cpp"):
        try:
            compiled_cache[p.name] = compile_tree(extract_tree(p.read_text(encoding="utf-8"), language="cpp"))
        except Exception:
            pass
            
    probs = []
    batch_size = 32
    valid_pairs = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            v_chunk = [p for p in chunk if p["a"] in compiled_cache and p["b"] in compiled_cache]
            if not v_chunk: continue
            valid_pairs.extend(v_chunk)
            ba = astnn_to_device(collate_trees([compiled_cache[p["a"]] for p in v_chunk]), DEVICE)
            bb = astnn_to_device(collate_trees([compiled_cache[p["b"]] for p in v_chunk]), DEVICE)
            logits = model(ba, bb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
            
    probs = np.array(probs)
    v_labels = np.array([p["label"] for p in valid_pairs])
    
    # Frozen th
    preds_frozen = (probs >= frozen_th).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(v_labels, preds_frozen).ravel()
    frozen_info = {
        'th': frozen_th, 'f1': f1_score(v_labels, preds_frozen, zero_division=0),
        'prec': precision_score(v_labels, preds_frozen, zero_division=0),
        'rec': recall_score(v_labels, preds_frozen, zero_division=0),
        'acc': accuracy_score(v_labels, preds_frozen),
        'tp': int(tp_f), 'fp': int(fp_f), 'tn': int(tn_f), 'fn': int(fn_f)
    }
    
    best_f1, best_th, best_info = -1, 0.5, None
    for th in np.linspace(0.01, 0.99, 100):
        preds = (probs >= th).astype(int)
        f1 = f1_score(v_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            tn, fp, fn, tp = confusion_matrix(v_labels, preds).ravel()
            best_info = {
                'th': th, 'f1': f1, 'prec': precision_score(v_labels, preds, zero_division=0),
                'rec': recall_score(v_labels, preds, zero_division=0),
                'acc': accuracy_score(v_labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    auc = roc_auc_score(v_labels, probs)
    print(f"[M3 ASTNN Fine-tuned BCB (Java)] ROC-AUC: {auc*100:.2f}%")
    print(f"  * Frozen th={frozen_info['th']:.2f}: F1={frozen_info['f1']*100:.2f}% | Prec={frozen_info['prec']*100:.2f}% | Rec={frozen_info['rec']*100:.2f}% | Acc={frozen_info['acc']*100:.2f}% | {frozen_info['tp']} / {frozen_info['fp']} / {frozen_info['tn']} / {frozen_info['fn']}")
    print(f"  * Best th={best_info['th']:.2f}: F1={best_info['f1']*100:.2f}% | Prec={best_info['prec']*100:.2f}% | Rec={best_info['rec']*100:.2f}% | Acc={best_info['acc']*100:.2f}% | {best_info['tp']} / {best_info['fp']} / {best_info['tn']} / {best_info['fn']}")
    return frozen_info, best_info

# --------------------------------------------------------------------------
# 4. M4 CPG-GNN Fine-tuned trên Java (BigCloneBench)
# --------------------------------------------------------------------------
def eval_cpg_bcb(ckpt_path: Path, frozen_th: float = 0.70):
    from src.models.cpg_gnn.model import CPGGNNPairClassifier
    from torch_geometric.data import Batch, Data
    from scripts.fast_native_cpg import FastNativeCPGExtractor
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    
    extractor = FastNativeCPGExtractor("cpp")
    graph_cache = {}
    for p in DATA_DIR.glob("*.cpp"):
        graph_cache[p.name] = extractor.parse_to_cpg(p.read_text(encoding="utf-8"))
        
    probs = []
    batch_size = 32
    valid_pairs = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            v_chunk = [p for p in chunk if p["a"] in graph_cache and p["b"] in graph_cache]
            if not v_chunk: continue
            valid_pairs.extend(v_chunk)
            ga = Batch.from_data_list([graph_cache[p["a"]] for p in v_chunk]).to(DEVICE)
            gb = Batch.from_data_list([graph_cache[p["b"]] for p in v_chunk]).to(DEVICE)
            logits = model(ga, gb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist() if p.ndim > 0 else [float(p)])
            
    probs = np.array(probs)
    v_labels = np.array([p["label"] for p in valid_pairs])
    
    preds_frozen = (probs >= frozen_th).astype(int)
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(v_labels, preds_frozen).ravel()
    frozen_info = {
        'th': frozen_th, 'f1': f1_score(v_labels, preds_frozen, zero_division=0),
        'prec': precision_score(v_labels, preds_frozen, zero_division=0),
        'rec': recall_score(v_labels, preds_frozen, zero_division=0),
        'acc': accuracy_score(v_labels, preds_frozen),
        'tp': int(tp_f), 'fp': int(fp_f), 'tn': int(tn_f), 'fn': int(fn_f)
    }
    
    best_f1, best_th, best_info = -1, 0.5, None
    for th in np.linspace(0.01, 0.99, 100):
        preds = (probs >= th).astype(int)
        f1 = f1_score(v_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            tn, fp, fn, tp = confusion_matrix(v_labels, preds).ravel()
            best_info = {
                'th': th, 'f1': f1, 'prec': precision_score(v_labels, preds, zero_division=0),
                'rec': recall_score(v_labels, preds, zero_division=0),
                'acc': accuracy_score(v_labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    auc = roc_auc_score(v_labels, probs)
    print(f"[M4 CPG-GNN Fine-tuned BCB (Java)] ROC-AUC: {auc*100:.2f}%")
    print(f"  * Frozen th={frozen_info['th']:.2f}: F1={frozen_info['f1']*100:.2f}% | Prec={frozen_info['prec']*100:.2f}% | Rec={frozen_info['rec']*100:.2f}% | Acc={frozen_info['acc']*100:.2f}% | {frozen_info['tp']} / {frozen_info['fp']} / {frozen_info['tn']} / {frozen_info['fn']}")
    print(f"  * Best th={best_info['th']:.2f}: F1={best_info['f1']*100:.2f}% | Prec={best_info['prec']*100:.2f}% | Rec={best_info['rec']*100:.2f}% | Acc={best_info['acc']*100:.2f}% | {best_info['tp']} / {best_info['fp']} / {best_info['tn']} / {best_info['fn']}")
    return frozen_info, best_info

# --------------------------------------------------------------------------
# Chạy toàn bộ
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("BẮT ĐẦU ĐÁNH GIÁ C/C++ CUSTOM SUITE BẰNG CÁC MÔ HÌNH HUẤN LUYỆN TRÊN JAVA (BCB)")
    print("=" * 80)
    
    # Zero-shot
    m1_zero = eval_zero_shot("microsoft/codebert-base", "M1 CodeBERT")
    m2_zero = eval_zero_shot("microsoft/graphcodebert-base", "M2 GraphCodeBERT")
    
    # Fine-tuned BCB (Java)
    m1_froz, m1_best, _ = eval_transformer_bcb(ROOT / "results" / "exp_bcb_002" / "best_checkpoint.pt", "microsoft/codebert-base", "M1 CodeBERT", 0.88)
    m2_froz, m2_best, _ = eval_transformer_bcb(ROOT / "results" / "exp_bcb_003" / "best_checkpoint.pt", "microsoft/graphcodebert-base", "M2 GraphCodeBERT", 0.94)
    m3_froz, m3_best = eval_astnn_bcb(ROOT / "results" / "exp_bcb_004" / "best_checkpoint.pt", 0.52)
    m4_froz, m4_best = eval_cpg_bcb(ROOT / "results" / "exp_bcb_005" / "best_checkpoint.pt", 0.70)
    
    # Tóm tắt
    out_data = {
        'M1_Zero': m1_zero,
        'M2_Zero': m2_zero,
        'M1_BCB_Frozen': m1_froz, 'M1_BCB_Best': m1_best,
        'M2_BCB_Frozen': m2_froz, 'M2_BCB_Best': m2_best,
        'M3_BCB_Frozen': m3_froz, 'M3_BCB_Best': m3_best,
        'M4_BCB_Frozen': m4_froz, 'M4_BCB_Best': m4_best,
    }
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2)
    print(f"\nĐã lưu toàn bộ kết quả vào: {OUT_PATH}")
