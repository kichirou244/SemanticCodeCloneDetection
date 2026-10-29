"""
Script: score_freeze_models.py
Chay danh gia toan dien cac checkpoint Freeze Backbone vua tai ve:
- results/M1_CodeBERT_Freeze/best_head.pt
- results/M2_GraphCodeBERT_Freeze/best_head.pt
tren ca 2 bo test:
- Java Custom Suite (720 cap)
- C/C++ Custom Suite (720 cap)

Xuat ket qua chi tiet ra:
- workspace/freeze_evaluation_results.json
Dong thoi tich hop vao workspace/custom_suite_java_scores.json va workspace/custom_suite_cpp_with_java_models_scores.json
"""

import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix

from src.models.transformer_pair import TransformerPairClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_suite(ckpt_path, hf_id, suite_dir, ext, model_name, suite_name):
    print(f"\n>>> Dang danh gia {model_name} ({ckpt_path}) tren {suite_name}...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = TransformerPairClassifier(hf_id=hf_id, freeze_backbone=True)
    model.head.load_state_dict(ckpt['head_state'])
    model.to(DEVICE)
    model.eval()

    manifest = Path(suite_dir) / 'pairs.jsonl'
    pairs = [json.loads(ln) for ln in manifest.read_text(encoding='utf-8').splitlines() if ln.strip()]
    files = {p.name: p.read_text(encoding='utf-8') for p in Path(suite_dir).glob(f'*.{ext}')}

    codes_a = [files[p['a']] for p in pairs]
    codes_b = [files[p['b']] for p in pairs]
    labels = np.array([p['label'] for p in pairs])

    probs = []
    logits_list = []
    pair_scores = {}
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            ca = codes_a[i : i + batch_size]
            cb = codes_b[i : i + batch_size]
            t1 = tokenizer(ca, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
            t2 = tokenizer(cb, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
            h1 = model.encode_side(t1)
            h2 = model.encode_side(t2)
            pair_repr = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)
            logits = model.head(pair_repr).squeeze(-1)
            p = torch.sigmoid(logits)

            p_cpu = p.cpu().numpy()
            l_cpu = logits.cpu().numpy()
            if p_cpu.ndim == 0:
                probs.append(float(p_cpu))
                logits_list.append(float(l_cpu))
            else:
                probs.extend(p_cpu.tolist())
                logits_list.extend(l_cpu.tolist())

    probs = np.array(probs)
    for p, prob in zip(pairs, probs):
        pair_scores[p['pair_id']] = float(prob)

    auc = roc_auc_score(labels, probs)

    # 1. Danh gia tai nguong mac dinh 0.5
    preds_def = (probs >= 0.5).astype(int)
    tn_d, fp_d, fn_d, tp_d = confusion_matrix(labels, preds_def).ravel()
    def_metrics = {
        'threshold': 0.5,
        'f1': float(f1_score(labels, preds_def, zero_division=0)),
        'precision': float(precision_score(labels, preds_def, zero_division=0)),
        'recall': float(recall_score(labels, preds_def, zero_division=0)),
        'accuracy': float(accuracy_score(labels, preds_def)),
        'tp': int(tp_d), 'fp': int(fp_d), 'tn': int(tn_d), 'fn': int(fn_d),
        'roc_auc': float(auc)
    }

    # 2. Quet nguong toi uu theo F1
    best_f1, best_th, best_preds, best_metrics = -1.0, 0.5, None, None
    for th in np.concatenate([np.logspace(-4, -0.01, 120), np.linspace(0.1, 0.99, 90)]):
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            best_preds = preds
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            best_metrics = {
                'threshold': float(th),
                'f1': float(f1),
                'precision': float(precision_score(labels, preds, zero_division=0)),
                'recall': float(recall_score(labels, preds, zero_division=0)),
                'accuracy': float(accuracy_score(labels, preds)),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                'roc_auc': float(auc)
            }

    # Breakdown chi tiet theo category tai nguong toi uu
    cat_stats = {}
    for p, pred in zip(pairs, best_preds):
        cat = p.get('category', 'unknown')
        lbl = p.get('label')
        if cat not in cat_stats:
            cat_stats[cat] = {'total': 0, 'correct': 0}
        cat_stats[cat]['total'] += 1
        if pred == lbl:
            cat_stats[cat]['correct'] += 1
    for cat, info in cat_stats.items():
        info['accuracy'] = info['correct'] / info['total']

    # Phan tich rieng trap
    trap_probs = [probs[i] for i, p in enumerate(pairs) if p.get('category') == 'nonclone_trap']
    trap_analysis = {
        'mean_prob': float(np.mean(trap_probs)) if trap_probs else 0.0,
        'max_prob': float(np.max(trap_probs)) if trap_probs else 0.0
    }

    print(f"  * ROC-AUC: {auc*100:.2f}%")
    print(f"  * Tai nguong def 0.50: F1={def_metrics['f1']*100:.2f}%, Prec={def_metrics['precision']*100:.2f}%, Rec={def_metrics['recall']*100:.2f}%, Acc={def_metrics['accuracy']*100:.2f}%")
    print(f"  * Tai nguong opt {best_metrics['threshold']:.4f}: F1={best_metrics['f1']*100:.2f}%, Prec={best_metrics['precision']*100:.2f}%, Rec={best_metrics['recall']*100:.2f}%, Acc={best_metrics['accuracy']*100:.2f}%")
    print(f"  * Nonclone Trap mean prob={trap_analysis['mean_prob']:.4f}, max={trap_analysis['max_prob']:.4f}")

    return {
        'model_name': model_name,
        'suite_name': suite_name,
        'default_metrics': def_metrics,
        'optimal_metrics': best_metrics,
        'by_category': cat_stats,
        'trap_analysis': trap_analysis,
        'pair_scores': pair_scores
    }

def main():
    m1_ckpt = ROOT / 'results' / 'M1_CodeBERT_Freeze' / 'best_head.pt'
    m2_ckpt = ROOT / 'results' / 'M2_GraphCodeBERT_Freeze' / 'best_head.pt'

    res = {}
    res['m2_java'] = evaluate_suite(m2_ckpt, 'microsoft/graphcodebert-base', ROOT / 'data' / 'custom_suite_java', 'java', 'M2_GraphCodeBERT_Freeze', 'Java')
    res['m2_cpp'] = evaluate_suite(m2_ckpt, 'microsoft/graphcodebert-base', ROOT / 'data' / 'custom_suite', 'cpp', 'M2_GraphCodeBERT_Freeze', 'C/C++')
    res['m1_java'] = evaluate_suite(m1_ckpt, 'microsoft/codebert-base', ROOT / 'data' / 'custom_suite_java', 'java', 'M1_CodeBERT_Freeze', 'Java')
    res['m1_cpp'] = evaluate_suite(m1_ckpt, 'microsoft/codebert-base', ROOT / 'data' / 'custom_suite', 'cpp', 'M1_CodeBERT_Freeze', 'C/C++')

    out_file = ROOT / 'workspace' / 'freeze_evaluation_results.json'
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n>>> Da luu toan bo ket qua vao: {out_file}")

    # Cap nhat vao custom_suite_java_scores.json
    java_scores_path = ROOT / 'workspace' / 'custom_suite_java_scores.json'
    if java_scores_path.exists():
        js_data = json.loads(java_scores_path.read_text(encoding='utf-8'))
        js_data['summary']['M1_Freeze'] = res['m1_java']['optimal_metrics']
        js_data['summary']['M1_Freeze']['by_category'] = res['m1_java']['by_category']
        js_data['summary']['M2_Freeze'] = res['m2_java']['optimal_metrics']
        js_data['summary']['M2_Freeze']['by_category'] = res['m2_java']['by_category']
        java_scores_path.write_text(json.dumps(js_data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f">>> Da cap nhat summary vao {java_scores_path}")

    # Cap nhat vao custom_suite_cpp_with_java_models_scores.json
    cpp_scores_path = ROOT / 'workspace' / 'custom_suite_cpp_with_java_models_scores.json'
    if cpp_scores_path.exists():
        cpp_data = json.loads(cpp_scores_path.read_text(encoding='utf-8'))
        cpp_data['M1_Freeze'] = res['m1_cpp']['optimal_metrics']
        cpp_data['M1_Freeze']['by_category'] = res['m1_cpp']['by_category']
        cpp_data['M2_Freeze'] = res['m2_cpp']['optimal_metrics']
        cpp_data['M2_Freeze']['by_category'] = res['m2_cpp']['by_category']
        cpp_scores_path.write_text(json.dumps(cpp_data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f">>> Da cap nhat summary vao {cpp_scores_path}")

if __name__ == '__main__':
    main()
