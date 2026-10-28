"""
Script: benchmark_zero_shot_untrained.py
Muc tieu:
Danh gia hieu nang cua 4 phuong phap o trang thai CHUA QUA HUAN LUYEN (Zero-shot / Pre-trained Raw):
- M1 (CodeBERT Base - microsoft/codebert-base) chua fine-tune, dung Cosine Similarity cua [CLS] vector.
- M2 (GraphCodeBERT Base - microsoft/graphcodebert-base) chua fine-tune, dung Cosine Similarity.
- M3 (ASTNN Untrained) kien truc RvNN + BiGRU khoi tao ban dau.
- M4 (CPG-GNN Untrained) kien truc GAT khoi tao ban dau.

Chay tren ca 2 tap kiem thu:
- Tap Java Custom Suite (720 cap, ti le 1:5)
- Tap C/C++ Custom Suite (720 cap, ti le 1:5)
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1
    }

def find_best_threshold(scores, labels):
    best_th = 0.5
    best_f1 = -1.0
    best_metrics = None
    for th in np.linspace(0.1, 0.99, 179):
        preds = [1 if s >= th else 0 for s in scores]
        m = compute_metrics(labels, preds)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m
    return best_th, best_metrics

def encode_transformers_zero_shot(model_id: str, files_dict: dict[str, str], device: torch.device):
    print(f"  * Nạp zero-shot model từ HuggingFace: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    
    embeddings = {}
    names = list(files_dict.keys())
    codes = [files_dict[k] for k in names]
    
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            chunk_codes = codes[i : i + batch_size]
            chunk_names = names[i : i + batch_size]
            
            inputs = tokenizer(
                chunk_codes, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(device)
            out = model(**inputs)
            # CLS embedding
            cls_emb = out.last_hidden_state[:, 0, :]
            # Normalize for cosine similarity
            norm_emb = F.normalize(cls_emb, p=2, dim=1).cpu()
            
            for name, emb in zip(chunk_names, norm_emb):
                embeddings[name] = emb
                
    return embeddings

def encode_astnn_untrained(files_dict: dict[str, str], language: str, device: torch.device):
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees
    
    n_stmt = 45 if language == "java" else 27
    torch.manual_seed(42)
    model = ASTNNPairClassifier(hidden=200, dropout=0.0, n_stmt=n_stmt).to(device)
    model.eval()
    
    compiled = {}
    for name, code in files_dict.items():
        compiled[name] = compile_tree(extract_tree(code, language=language))
        
    embeddings = {}
    names = list(files_dict.keys())
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            chunk = names[i : i + batch_size]
            batch_compiled = [compiled[n] for n in chunk]
            batch_trees = astnn_to_device(collate_trees(batch_compiled), device)
            # Encode qua encode_side
            h = model.encode_side(batch_trees)
            norm_h = F.normalize(h, p=2, dim=1).cpu()
            for name, emb in zip(chunk, norm_h):
                embeddings[name] = emb
    return embeddings

def encode_cpg_untrained(files_dict: dict[str, str], language: str, device: torch.device):
    from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device as cpg_to_device
    from torch_geometric.data import Batch, Data
    from src.representations.ast.extract import extract_tree
    
    torch.manual_seed(42)
    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.0).to(device)
    model.eval()
    
    def code_to_data(code: str) -> Data:
        tree = extract_tree(code, language=language)
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
        node_type = torch.zeros(n, dtype=torch.long)
        x = torch.zeros((n, 1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_type=edge_type, node_type=node_type, num_nodes=n)
        
    graphs = {name: code_to_data(code) for name, code in files_dict.items()}
    names = list(files_dict.keys())
    embeddings = {}
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            chunk = names[i : i + batch_size]
            batch_data = cpg_to_device(Batch.from_data_list([graphs[n] for n in chunk]), device)
            h = model.encoder(batch_data)
            norm_h = F.normalize(h, p=2, dim=1).cpu()
            for name, emb in zip(chunk, norm_h):
                embeddings[name] = emb
    return embeddings

def run_zero_shot_suite(suite_dir: Path, ext: str, language: str):
    suite_name = "JAVA" if language == "java" else "C/C++"
    print("=" * 80)
    print(f"DANH GIA KHONG HUAN LUYEN (ZERO-SHOT / PRE-TRAINED RAW): {suite_name} (720 CAP, 1:5)")
    print("=" * 80)
    
    manifest_path = suite_dir / "pairs.jsonl"
    pairs = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [p["label"] for p in pairs]
    cats = [p["category"] for p in pairs]
    
    files_dict = {p.name: p.read_text(encoding="utf-8") for p in suite_dir.glob(f"*.{ext}")}
    print(f"Tổng số tệp mã nguồn: {len(files_dict)} files, Tổng số cặp: {len(pairs)} pairs")
    
    # 1. M1 CodeBERT Zero-shot
    emb_m1 = encode_transformers_zero_shot("microsoft/codebert-base", files_dict, DEVICE)
    scores_m1 = [(torch.dot(emb_m1[p["a"]], emb_m1[p["b"]]).item() + 1.0) / 2.0 for p in pairs] # map [-1,1] to [0,1]
    
    # 2. M2 GraphCodeBERT Zero-shot
    emb_m2 = encode_transformers_zero_shot("microsoft/graphcodebert-base", files_dict, DEVICE)
    scores_m2 = [(torch.dot(emb_m2[p["a"]], emb_m2[p["b"]]).item() + 1.0) / 2.0 for p in pairs]
    
    # 3. M3 ASTNN Untrained
    print("  * Nạp M3 (ASTNN Untrained)...")
    emb_m3 = encode_astnn_untrained(files_dict, language, DEVICE)
    scores_m3 = [(torch.dot(emb_m3[p["a"]], emb_m3[p["b"]]).item() + 1.0) / 2.0 for p in pairs]
    
    # 4. M4 CPG-GNN Untrained
    print("  * Nạp M4 (CPG-GNN Untrained)...")
    emb_m4 = encode_cpg_untrained(files_dict, language, DEVICE)
    scores_m4 = [(torch.dot(emb_m4[p["a"]], emb_m4[p["b"]]).item() + 1.0) / 2.0 for p in pairs]
    
    models = {
        "M1 (CodeBERT Raw Zero-shot)": scores_m1,
        "M2 (GraphCodeBERT Raw Zero-shot)": scores_m2,
        "M3 (ASTNN Untrained)": scores_m3,
        "M4 (CPG-GNN Untrained)": scores_m4,
    }
    
    results = {}
    print("\nKET QUA (Zero-shot / Chua qua fine-tuning):")
    print(f"{'Mô hình':<35} | {'Ngưỡng':<7} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'Accuracy':<9}")
    print("-" * 85)
    for name, s in models.items():
        th, m = find_best_threshold(s, labels)
        results[name] = {"th": th, "metrics": m, "scores": s}
        print(f"{name:<35} | {th:<7.2f} | {m['f1']*100:<7.2f}% | {m['precision']*100:<9.2f}% | {m['recall']*100:<7.2f}% | {m['accuracy']*100:<8.2f}%")
        
    return results

if __name__ == "__main__":
    res_java = run_zero_shot_suite(ROOT / "data" / "custom_suite_java", "java", "java")
    print("\n\n")
    res_cpp = run_zero_shot_suite(ROOT / "data" / "custom_suite", "cpp", "cpp")
    
    # Lưu kết quả
    out_file = ROOT / "workspace" / "zero_shot_benchmark_scores.json"
    clean_out = {}
    for lang, res in [("java", res_java), ("cpp", res_cpp)]:
        clean_out[lang] = {
            m: {"th": d["th"], "metrics": d["metrics"]}
            for m, d in res.items()
        }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_out, f, indent=2)
    print(f"\nĐã lưu kết quả Zero-shot vào: {out_file}")
