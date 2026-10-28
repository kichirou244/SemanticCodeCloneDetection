"""
Script: optimize_thresholds_and_calibration.py
Muc tieu:
1. Trien khai Bayesian Logit Shift Calibration de bu tru ty le mat can bang 1:5.
2. Toi uu hoa nguong theo do do F0.5 (coi trong Precision hon Recall gap 2 lan).
3. Trien khai Cascade / Dual-Threshold Ensembling ket hop M1 (CodeBERT) va M2 (GraphCodeBERT).
"""

import json
import math
import numpy as np
from pathlib import Path

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    beta = 0.5
    f05 = (1 + beta**2) * prec * rec / ((beta**2 * prec) + rec) if ((beta**2 * prec) + rec) > 0 else 0.0
    
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "f05": f05
    }

def evaluate_threshold(scores, labels, threshold):
    preds = [1 if s >= threshold else 0 for s in scores]
    return compute_metrics(labels, preds)

def find_optimal_threshold(scores, labels, metric="f1"):
    best_th = 0.5
    best_val = -1.0
    best_metrics = None
    
    for th in np.linspace(0.01, 0.99, 197):
        m = evaluate_threshold(scores, labels, th)
        val = m[metric]
        if val > best_val:
            best_val = val
            best_th = th
            best_metrics = m
            
    return best_th, best_metrics

def run_suite_analysis(json_path, suite_name="Java"):
    print("=" * 80)
    print(f"PHAN TICH TOI UU HOA NGUONG & CALIBRATION: {suite_name.upper()} (TỈ LỆ 1:5)")
    print("=" * 80)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    pairs = data["pairs_detail"]
    labels = [p["label"] for p in pairs]
    cats = [p["category"] for p in pairs]
    
    models = ["M1", "M2", "M3", "M4"]
    results = {}
    
    for m in models:
        scores = np.array([p["scores"][m] for p in pairs])
        baseline_th = data["summary"][m]["threshold"]
        baseline_res = evaluate_threshold(scores, labels, baseline_th)
        
        # 1. Optimal F1 Threshold
        th_opt_f1, res_opt_f1 = find_optimal_threshold(scores, labels, metric="f1")
        
        # 2. Optimal F0.5 Threshold (Precision-focused)
        th_opt_f05, res_opt_f05 = find_optimal_threshold(scores, labels, metric="f05")
        
        # 3. Bayesian Logit Shift Calibration (Prior odds shift: ln(1/5) - ln(1/1) = -ln(5))
        prior_shift = -np.log(5.0)
        z = logit(scores)
        calibrated_scores = sigmoid(z + prior_shift)
        th_cal = 0.5 # Standard neutral threshold on calibrated probability
        res_cal = evaluate_threshold(calibrated_scores, labels, th_cal)
        
        # Breakdown on trap non-clones
        trap_indices = [i for i, c in enumerate(cats) if c == "nonclone_trap"]
        trap_labels = [labels[i] for i in trap_indices]
        
        trap_acc_base = (evaluate_threshold([scores[i] for i in trap_indices], trap_labels, baseline_th))["accuracy"]
        trap_acc_f05 = (evaluate_threshold([scores[i] for i in trap_indices], trap_labels, th_opt_f05))["accuracy"]
        trap_acc_cal = (evaluate_threshold([calibrated_scores[i] for i in trap_indices], trap_labels, th_cal))["accuracy"]
        
        results[m] = {
            "baseline": (baseline_th, baseline_res, trap_acc_base),
            "opt_f1": (th_opt_f1, res_opt_f1),
            "opt_f05": (th_opt_f05, res_opt_f05, trap_acc_f05),
            "calibrated": (th_cal, res_cal, trap_acc_cal)
        }
        
        print(f"\n[{m}] (Default th = {baseline_th:.2f}):")
        print(f"  * Baseline      : F1={baseline_res['f1']*100:.2f}%, Prec={baseline_res['precision']*100:.2f}%, Rec={baseline_res['recall']*100:.2f}%, Acc={baseline_res['accuracy']*100:.2f}%, Trap-TN={trap_acc_base*100:.1f}%")
        print(f"  * Opt-F1 (th={th_opt_f1:.2f}): F1={res_opt_f1['f1']*100:.2f}%, Prec={res_opt_f1['precision']*100:.2f}%, Rec={res_opt_f1['recall']*100:.2f}%, Acc={res_opt_f1['accuracy']*100:.2f}%")
        print(f"  * Opt-F0.5(th={th_opt_f05:.2f}): F1={res_opt_f05['f1']*100:.2f}%, Prec={res_opt_f05['precision']*100:.2f}%, Rec={res_opt_f05['recall']*100:.2f}%, Acc={res_opt_f05['accuracy']*100:.2f}%, Trap-TN={trap_acc_f05*100:.1f}%")
        print(f"  * Bayes-Calib   : F1={res_cal['f1']*100:.2f}%, Prec={res_cal['precision']*100:.2f}%, Rec={res_cal['recall']*100:.2f}%, Acc={res_cal['accuracy']*100:.2f}%, Trap-TN={trap_acc_cal*100:.1f}%")

    # 4. CASCADE / DUAL-THRESHOLD ENSEMBLE: M1 (high recall) + M2 (high precision)
    scores_m1 = np.array([p["scores"]["M1"] for p in pairs])
    scores_m2 = np.array([p["scores"]["M2"] for p in pairs])
    
    # Simple Cascade:
    # If M1 thinks it's non-clone (< th_m1_low), accept 0.
    # If M1 >= th_m1_low, defer to M2!
    th_m1_low = 0.80
    th_m2 = results["M2"]["opt_f05"][0]
    
    cascade_preds = []
    for s1, s2 in zip(scores_m1, scores_m2):
        if s1 < th_m1_low:
            cascade_preds.append(0)
        else:
            cascade_preds.append(1 if s2 >= th_m2 else 0)
            
    cascade_res = compute_metrics(labels, cascade_preds)
    trap_acc_cascade = evaluate_threshold([cascade_preds[i] for i in trap_indices], trap_labels, 0.5)["accuracy"]
    
    print("\n" + "-" * 50)
    print("[CASCADE ENSEMBLE: M1 -> M2 (Bảo vệ đa tầng)]")
    print(f"  * M1 sơ lọc (s1 < {th_m1_low}) + M2 phán quyết (s2 >= {th_m2:.2f}):")
    print(f"    F1={cascade_res['f1']*100:.2f}%, Prec={cascade_res['precision']*100:.2f}%, Rec={cascade_res['recall']*100:.2f}%, Acc={cascade_res['accuracy']*100:.2f}%, Trap-TN={trap_acc_cascade*100:.1f}%")

    return results

if __name__ == "__main__":
    res_java = run_suite_analysis("workspace/custom_suite_java_scores.json", "Java")
    print("\n\n")
    res_cpp = run_suite_analysis("workspace/custom_suite_cpp_scores.json", "C/C++")
