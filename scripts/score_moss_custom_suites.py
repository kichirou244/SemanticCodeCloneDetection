"""
Script: score_moss_custom_suites.py
Chay baseline B0/B1 - Stanford MOSS that tren ca 2 bo test suite:
- Java Custom Suite (90 files, 720 cap)
- C/C++ Custom Suite (121 files, 720 cap)
Goi truc tiep service moss.stanford.edu qua mosspy / MossServiceClient,
doc ket qua HTML tra ve, trich xuat score va tinh toan F1, Precision, Recall, Accuracy, TP, FP, TN, FN.
"""

import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baseline.moss_service import MossServiceClient

def run_moss_on_suite(suite_dir: Path, ext: str, lang: str, suite_name: str, thr: float = 0.01):
    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY MOSS SERVICE STANFORD TRÊN: {suite_name} (Language={lang})")
    print("=" * 80)
    
    files = {p.name: p.read_text(encoding='utf-8').replace("\x00", " ") for p in suite_dir.glob(f"*.{ext}")}
    print(f"Số lượng file mã nguồn nộp lên Stanford MOSS: {len(files)}")
    
    client = MossServiceClient(language=lang, show=1000, max_matches=1000)
    report = client.submit(files)
    print(f"URL báo cáo MOSS Stanford: {report.url}")
    print(f"Số lượng cặp được MOSS phát hiện trùng lặp: {len(report.matches)}")
    
    manifest = suite_dir / "pairs.jsonl"
    pairs = [json.loads(ln) for ln in manifest.read_text(encoding='utf-8').splitlines() if ln.strip()]
    labels = np.array([p['label'] for p in pairs])
    
    scores = []
    for p in pairs:
        # Score = max(pct_a, pct_b) / 100
        sc = report.score(p['a'], p['b'])
        scores.append(sc)
    scores = np.array(scores)
    
    # 1. Tại rule vận hành chuẩn của MOSS: score > 0.01 (MOSS gắn cờ = clone)
    preds_standard = (scores > thr).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(labels, preds_standard).ravel()
    metrics_standard = {
        'url': report.url,
        'thr': thr,
        'f1': f1_score(labels, preds_standard, zero_division=0),
        'prec': precision_score(labels, preds_standard, zero_division=0),
        'rec': recall_score(labels, preds_standard, zero_division=0),
        'acc': accuracy_score(labels, preds_standard),
        'tp': int(tp_s), 'fp': int(fp_s), 'tn': int(tn_s), 'fn': int(fn_s)
    }
    print(f"[MOSS {suite_name} - Standard (thr > {thr})]")
    print(f"  F1: {metrics_standard['f1']*100:.2f}% | Prec: {metrics_standard['prec']*100:.2f}% | Rec: {metrics_standard['rec']*100:.2f}% | Acc: {metrics_standard['acc']*100:.2f}%")
    print(f"  TP / FP / TN / FN: {metrics_standard['tp']} / {metrics_standard['fp']} / {metrics_standard['tn']} / {metrics_standard['fn']}")
    
    # 2. Optimal threshold search
    best_f1, best_th, best_info = -1, 0.01, None
    for t in np.linspace(0.001, 0.99, 100):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = t
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            best_info = {
                'thr': t,
                'f1': f1,
                'prec': precision_score(labels, preds, zero_division=0),
                'rec': recall_score(labels, preds, zero_division=0),
                'acc': accuracy_score(labels, preds),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            }
    print(f"[MOSS {suite_name} - Best Threshold (thr >= {best_info['thr']:.2f})]")
    print(f"  F1: {best_info['f1']*100:.2f}% | Prec: {best_info['prec']*100:.2f}% | Rec: {best_info['rec']*100:.2f}% | Acc: {best_info['acc']*100:.2f}%")
    print(f"  TP / FP / TN / FN: {best_info['tp']} / {best_info['fp']} / {best_info['tn']} / {best_info['fn']}")
    
    # Category breakdown
    cat_counts = {}
    for p, pred in zip(pairs, preds_standard):
        c = p.get('category', 'unknown')
        lbl = p.get('label')
        if c not in cat_counts: cat_counts[c] = {'tot': 0, 'cor': 0}
        cat_counts[c]['tot'] += 1
        if pred == lbl: cat_counts[c]['cor'] += 1
    cat_res = {k: f"{v['cor']/v['tot']*100:.1f}%" for k, v in cat_counts.items()}
    print(f"Chi tiết theo category: {cat_res}")
    
    return metrics_standard, best_info, cat_res

if __name__ == "__main__":
    res_java_std, res_java_best, cat_java = run_moss_on_suite(ROOT / "data" / "custom_suite_java", "java", "java", "Java Custom Suite")
    res_cpp_std, res_cpp_best, cat_cpp = run_moss_on_suite(ROOT / "data" / "custom_suite", "cpp", "cc", "C/C++ Custom Suite")
    
    out_all = {
        'java_standard': res_java_std,
        'java_best': res_java_best,
        'java_categories': cat_java,
        'cpp_standard': res_cpp_std,
        'cpp_best': res_cpp_best,
        'cpp_categories': cat_cpp
    }
    with open(ROOT / "workspace" / "moss_custom_suite_scores.json", "w", encoding="utf-8") as f:
        json.dump(out_all, f, indent=2)
    print(f"\nĐã lưu kết quả MOSS vào: {ROOT / 'workspace' / 'moss_custom_suite_scores.json'}")
