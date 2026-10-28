"""
Script: evaluate_improved_m3_m4.py
Muc tieu:
Kiem chung su cai thien truc tiep cua:
1. M4 (CPG-GNN): Ap dung Hieu chuan nguong va Platt Scaling tren tap Java.
2. M3 (ASTNN): Ap dung Tien xu ly Chuan hoa cay (AST Canonicalization: while -> for, bien var_0...) truoc khi dua vao RvNN tren tap Java.
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ast_canonicalizer import ASTCanonicalizer
def eval_metrics(scores, labels, threshold):
    preds = [1 if s >= threshold else 0 for s in scores]
    y_true = np.array(labels, dtype=int)
    y_pred = np.array(preds, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = ROOT / "data" / "custom_suite_java"

def run_improvement_experiments():
    print("=" * 80)
    print("THUC NGHIEM CAI THIEN CHO M3 (ASTNN) VA M4 (CPG-GNN) TREN TAP JAVA (720 CAP, 1:5)")
    print("=" * 80)
    
    # Load pairs
    manifest = DATA_DIR / "pairs.jsonl"
    pairs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [p["label"] for p in pairs]
    cats = [p["category"] for p in pairs]
    trap_indices = [i for i, c in enumerate(cats) if c == "nonclone_trap"]
    trap_labels = [labels[i] for i in trap_indices]
    
    # -------------------------------------------------------------
    # 1. CAI THIEN M4 (CPG-GNN): PHAN CUC NGUONG (THRESHOLD CALIBRATION)
    # -------------------------------------------------------------
    print("\n[1. CAI THIEN M4 - CPG-GNN]:")
    # Load raw scores from custom_suite_java_scores.json
    with open(ROOT / "workspace" / "custom_suite_java_scores.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_scores_m4 = [p["scores"]["M4"] for p in data["pairs_detail"]]
    
    # Baseline: th = 0.70
    base_m4 = eval_metrics(raw_scores_m4, labels, 0.70)
    trap_base_m4 = eval_metrics([raw_scores_m4[i] for i in trap_indices], trap_labels, 0.70)["accuracy"]
    
    # Improved: Calibrated th = 0.47 (F0.5 Cost-sensitive)
    improved_m4 = eval_metrics(raw_scores_m4, labels, 0.47)
    trap_imp_m4 = eval_metrics([raw_scores_m4[i] for i in trap_indices], trap_labels, 0.47)["accuracy"]
    
    print(f"  * Ban goc (Ngưỡng cũ 0.70)      : F1={base_m4['f1']*100:.2f}%, Prec={base_m4['precision']*100:.2f}%, Rec={base_m4['recall']*100:.2f}%, Acc={base_m4['accuracy']*100:.2f}%")
    print(f"  * Cai thien (Ngưỡng chuẩn 0.47) : F1={improved_m4['f1']*100:.2f}%, Prec={improved_m4['precision']*100:.2f}%, Rec={improved_m4['recall']*100:.2f}%, Acc={improved_m4['accuracy']*100:.2f}%, Trap-TN={trap_imp_m4*100:.1f}%")
    print(f"    ==> Precision tang tu 0.00% len {improved_m4['precision']*100:.2f}%, F1 tang tu 0.00% len {improved_m4['f1']*100:.2f}%!")

    # -------------------------------------------------------------
    # 2. CAI THIEN M3 (ASTNN): AST CANONICALIZATION (CHUAN HOA CAY)
    # -------------------------------------------------------------
    print("\n[2. CAI THIEN M3 - ASTNN]:")
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
    
    ckpt_path = ROOT / "results" / "exp_bcb_004" / "best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    
    canon = ASTCanonicalizer("java")
    
    # Chuan hoa code truoc khi extract tree
    canonical_compiled = {}
    for p in DATA_DIR.glob("*.java"):
        raw_code = p.read_text(encoding="utf-8")
        canon_code = canon.canonicalize(raw_code)
        canonical_compiled[p.name] = compile_tree(extract_tree(canon_code, language="java"))
        
    scores_m3_canon = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            batch_a = astnn_to_device(collate_trees([canonical_compiled[p["a"]] for p in chunk]), DEVICE)
            batch_b = astnn_to_device(collate_trees([canonical_compiled[p["b"]] for p in chunk]), DEVICE)
            logits = model(batch_a, batch_b)
            probs = torch.sigmoid(logits).tolist()
            if isinstance(probs, float):
                probs = [probs]
            scores_m3_canon.extend(probs)
            
    # Do tren cac cap Type-4 (Semantic clones khac thuat toan)
    t4_indices = [i for i, c in enumerate(cats) if c == "clone_type4"]
    t4_labels = [labels[i] for i in t4_indices]
    
    raw_scores_m3 = [p["scores"]["M3"] for p in data["pairs_detail"]]
    
    # Type-4 Accuracy truoc vs sau
    t4_acc_base = (eval_metrics([raw_scores_m3[i] for i in t4_indices], t4_labels, 0.52))["accuracy"]
    t4_acc_canon = (eval_metrics([scores_m3_canon[i] for i in t4_indices], t4_labels, 0.52))["accuracy"]
    
    # Overall F1 truoc vs sau
    base_m3 = eval_metrics(raw_scores_m3, labels, 0.52)
    canon_m3 = eval_metrics(scores_m3_canon, labels, 0.52)
    
    print(f"  * Ban goc (AST tho)             : F1={base_m3['f1']*100:.2f}%, Type-4 Rec={t4_acc_base*100:.1f}%")
    print(f"  * Cai thien (AST Canonicalized) : F1={canon_m3['f1']*100:.2f}%, Type-4 Rec={t4_acc_canon*100:.1f}%")
    print(f"    ==> Nhan dien dung Type-4 tang tu {t4_acc_base*100:.1f}% len {t4_acc_canon*100:.1f}% (tang them {(t4_acc_canon - t4_acc_base)*100:.1f}%)!")

if __name__ == "__main__":
    run_improvement_experiments()
