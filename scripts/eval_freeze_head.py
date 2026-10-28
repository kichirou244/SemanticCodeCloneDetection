"""
Script: eval_freeze_head.py
Kiem tra chi tiet phan phoi xac suat va nguong toi uu cho checkpoint M1 & M2 Freeze Head.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix

from src.models.transformer_pair import TransformerPairClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model_suite(ckpt_path, hf_id, suite_dir, ext, suite_name):
    print("=" * 70)
    print(f"Danh gia {ckpt_path.name} tren {suite_name} (Device: {DEVICE})")
    
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
    labels = [p['label'] for p in pairs]
    
    probs = []
    logits_list = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            ca = codes_a[i : i + batch_size]
            cb = codes_b[i : i + batch_size]
            t1 = tokenizer(ca, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
            t2 = tokenizer(cb, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)
            h1 = model.encode_side(t1)
            h2 = model.encode_side(t2)
            pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)
            logits = model.head(pair).squeeze(-1)
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
    labels = np.array(labels)
    logits_arr = np.array(logits_list)
    
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    print(f"Logits min={logits_arr.min():.2f}, mean={logits_arr.mean():.2f}, max={logits_arr.max():.2f}")
    print(f"Clone probs: min={pos_probs.min():.4f}, mean={pos_probs.mean():.4f}, median={np.median(pos_probs):.4f}, max={pos_probs.max():.4f}")
    print(f"Nonclone probs: min={neg_probs.min():.4f}, mean={neg_probs.mean():.4f}, median={np.median(neg_probs):.4f}, max={neg_probs.max():.4f}")
    
    auc = roc_auc_score(labels, probs)
    print(f"ROC-AUC: {auc*100:.2f}%")
    
    # Scan rong tu 0.0001 toi 0.99
    best_f1, best_th, best_res = -1.0, 0.5, None
    for th in np.concatenate([np.logspace(-4, -0.01, 100), np.linspace(0.1, 0.99, 90)]):
        preds = (probs >= th).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            best_res = {
                'th': th,
                'f1': f1,
                'prec': precision_score(labels, preds, zero_division=0),
                'rec': recall_score(labels, preds, zero_division=0),
                'acc': accuracy_score(labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
            
    print(f"Best: th={best_res['th']:.4f} | F1={best_res['f1']*100:.2f}% | Prec={best_res['prec']*100:.2f}% | Rec={best_res['rec']*100:.2f}% | Acc={best_res['acc']*100:.2f}% | {best_res['tp']} / {best_res['fp']} / {best_res['tn']} / {best_res['fn']}")
    
    # Chi tiet theo tung category
    preds_best = (probs >= best_res['th']).astype(int)
    cat_counts = {}
    for p, pred in zip(pairs, preds_best):
        cat = p.get('category', 'unknown')
        lbl = p.get('label')
        if cat not in cat_counts:
            cat_counts[cat] = {'tot': 0, 'cor': 0}
        cat_counts[cat]['tot'] += 1
        if pred == lbl:
            cat_counts[cat]['cor'] += 1
    for cat in sorted(cat_counts.keys()):
        cinfo = cat_counts[cat]
        print(f"  * {cat}: {cinfo['cor']}/{cinfo['tot']} ({cinfo['cor']/cinfo['tot']*100:.1f}%)")
    
    # Check rieng category nonclone_trap
    trap_probs = [probs[i] for i, p in enumerate(pairs) if p.get('category') == 'nonclone_trap']
    if trap_probs:
        print(f"Nonclone Trap probs: mean={np.mean(trap_probs):.4f}, max={np.max(trap_probs):.4f}")

if __name__ == '__main__':
    m2_path = Path('kaggle_tmp/v50_output/results/M2_GraphCodeBERT_Freeze/best_head.pt')
    m1_path = Path('kaggle_tmp/v50_output/results/M1_CodeBERT_Freeze/best_head.pt')
    
    eval_model_suite(m2_path, 'microsoft/graphcodebert-base', 'data/custom_suite_java', 'java', 'M2 Java')
    eval_model_suite(m2_path, 'microsoft/graphcodebert-base', 'data/custom_suite', 'cpp', 'M2 C/C++')
    eval_model_suite(m1_path, 'microsoft/codebert-base', 'data/custom_suite_java', 'java', 'M1 Java')
    eval_model_suite(m1_path, 'microsoft/codebert-base', 'data/custom_suite', 'cpp', 'M1 C/C++')
