#!/usr/bin/env python3
"""Huan luyen 4 phuong phap (M1 CodeBERT, M2 GraphCodeBERT, M3 ASTNN, M4 CPG-GNN)
tren 500 mau Train (50:50) va danh gia tren 500 mau Test (50:50).
Ghi nhan Loss va Accuracy qua tung epoch (3 epoch cho M1, M2, M3 va 5 epoch cho M4),
ve bieu do so sanh Loss/Acc giua Model 500 mau vs Model ban dau (Full Train).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = ROOT / "results" / "bcb_500_train_pairs.jsonl"
TEST_FILE = ROOT / "results" / "bcb_500_test_pairs.jsonl"
OUT_RESULTS = ROOT / "results" / "train_500_models_comparison.json"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_pairs(filepath):
    pairs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc * 100, "prec": prec * 100, "rec": rec * 100, "f1": f1 * 100}


# ==========================================
# 1. TRAIN M1 (CodeBERT) & M2 (GraphCodeBERT)
# ==========================================
class PairHead(nn.Module):
    def __init__(self, in_dim=768 * 3, hidden_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h1, h2):
        pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)
        return self.net(pair).squeeze(-1)


def train_transformer_head(hf_id, model_name, train_pairs, test_pairs, epochs=3):
    print(f"\n===== Huan luyen {model_name} tren 500 mau Train ({epochs} epochs) =====")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    backbone = AutoModel.from_pretrained(hf_id).to(DEVICE)
    backbone.eval()

    def extract(pairs, bs=64):
        c1 = [p["func1"] for p in pairs]
        c2 = [p["func2"] for p in pairs]
        y = torch.tensor([p["label"] for p in pairs], dtype=torch.float)
        h1_l, h2_l = [], []
        with torch.no_grad():
            for i in range(0, len(pairs), bs):
                b1 = tokenizer(c1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                b2 = tokenizer(c2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
                h1_l.append(backbone(**b1).last_hidden_state[:, 0, :].cpu())
                h2_l.append(backbone(**b2).last_hidden_state[:, 0, :].cpu())
        return torch.cat(h1_l, dim=0), torch.cat(h2_l, dim=0), y

    print(f"[{model_name}] Trich xuat features Train & Test...")
    tr_h1, tr_h2, tr_y = extract(train_pairs)
    te_h1, te_h2, te_y = extract(test_pairs)
    del backbone
    torch.cuda.empty_cache()

    head = PairHead().to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(torch.utils.data.TensorDataset(tr_h1, tr_h2, tr_y), batch_size=32, shuffle=True)

    history = []
    for ep in range(1, epochs + 1):
        head.train()
        total_tr_loss = 0.0
        for b1, b2, by in train_loader:
            b1, b2, by = b1.to(DEVICE), b2.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = head(b1, b2)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_tr_loss += loss.item() * len(by)
        train_loss = total_tr_loss / len(tr_y)

        # Eval
        head.eval()
        with torch.no_grad():
            # Train acc
            tr_logits = head(tr_h1.to(DEVICE), tr_h2.to(DEVICE))
            tr_preds = (torch.sigmoid(tr_logits) >= 0.5).cpu().numpy().astype(int)
            tr_m = compute_metrics(tr_y.numpy(), tr_preds)

            # Test loss & metrics
            te_logits = head(te_h1.to(DEVICE), te_h2.to(DEVICE))
            test_loss = criterion(te_logits, te_y.to(DEVICE)).item()
            te_preds = (torch.sigmoid(te_logits) >= 0.5).cpu().numpy().astype(int)
            te_m = compute_metrics(te_y.numpy(), te_preds)

        print(f"[{model_name}] Epoch {ep}: Train Loss={train_loss:.4f}, Train Acc={tr_m['acc']:.2f}% | Test Loss={test_loss:.4f}, Test Acc={te_m['acc']:.2f}%, Test F1={te_m['f1']:.2f}%")
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_acc": round(tr_m["acc"], 2),
            "test_acc": round(te_m["acc"], 2),
            "test_f1": round(te_m["f1"], 2),
            "test_prec": round(te_m["prec"], 2),
            "test_rec": round(te_m["rec"], 2),
        })

    return history, history[-1]


# ==========================================
# 2. TRAIN M3 (ASTNN)
# ==========================================
def train_astnn(train_pairs, test_pairs, epochs=3):
    print(f"\n===== Huan luyen M3 (ASTNN) tren 500 mau Train ({epochs} epochs) =====")
    from src.models.astnn.model import ASTNNPairClassifier, to_device as astnn_to_device
    from src.representations.ast.extract import compile_tree, extract_tree, collate_trees

    print("[M3] Compile AST cho 500 Train va 500 Test...")
    tr_c1 = [compile_tree(extract_tree(p["func1"], language="java")) for p in train_pairs]
    tr_c2 = [compile_tree(extract_tree(p["func2"], language="java")) for p in train_pairs]
    tr_y = [p["label"] for p in train_pairs]

    te_c1 = [compile_tree(extract_tree(p["func1"], language="java")) for p in test_pairs]
    te_c2 = [compile_tree(extract_tree(p["func2"], language="java")) for p in test_pairs]
    te_y = [p["label"] for p in test_pairs]

    model = ASTNNPairClassifier(hidden=200, dropout=0.2, n_stmt=45).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    bs = 32
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        # Shuffle train
        idxs = list(range(len(train_pairs)))
        random.shuffle(idxs)
        for i in range(0, len(idxs), bs):
            chunk = idxs[i : i + bs]
            b1 = astnn_to_device(collate_trees([tr_c1[j] for j in chunk]), DEVICE)
            b2 = astnn_to_device(collate_trees([tr_c2[j] for j in chunk]), DEVICE)
            by = torch.tensor([tr_y[j] for j in chunk], dtype=torch.float, device=DEVICE)

            optimizer.zero_grad()
            logits = model(b1, b2)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(chunk)
        train_loss = total_loss / len(train_pairs)

        # Eval
        model.eval()
        with torch.no_grad():
            # Eval on Train
            tr_preds = []
            for i in range(0, len(train_pairs), bs):
                b1 = astnn_to_device(collate_trees(tr_c1[i : i + bs]), DEVICE)
                b2 = astnn_to_device(collate_trees(tr_c2[i : i + bs]), DEVICE)
                probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
                tr_preds.extend([1 if p >= 0.5 else 0 for p in probs])
            tr_m = compute_metrics(tr_y, tr_preds)

            # Eval on Test
            te_preds, te_losses = [], []
            for i in range(0, len(test_pairs), bs):
                b1 = astnn_to_device(collate_trees(te_c1[i : i + bs]), DEVICE)
                b2 = astnn_to_device(collate_trees(te_c2[i : i + bs]), DEVICE)
                by = torch.tensor(te_y[i : i + bs], dtype=torch.float, device=DEVICE)
                logits = model(b1, b2)
                te_losses.append(criterion(logits, by).item() * len(by))
                probs = torch.sigmoid(logits).cpu().tolist()
                te_preds.extend([1 if p >= 0.5 else 0 for p in probs])
            test_loss = sum(te_losses) / len(test_pairs)
            te_m = compute_metrics(te_y, te_preds)

        print(f"[M3] Epoch {ep}: Train Loss={train_loss:.4f}, Train Acc={tr_m['acc']:.2f}% | Test Loss={test_loss:.4f}, Test Acc={te_m['acc']:.2f}%, Test F1={te_m['f1']:.2f}%")
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_acc": round(tr_m["acc"], 2),
            "test_acc": round(te_m["acc"], 2),
            "test_f1": round(te_m["f1"], 2),
            "test_prec": round(te_m["prec"], 2),
            "test_rec": round(te_m["rec"], 2),
        })

    return history, history[-1]


# ==========================================
# 3. TRAIN M4 (CPG-GNN)
# ==========================================
def train_cpg(train_pairs, test_pairs, epochs=5):
    print(f"\n===== Huan luyen M4 (CPG-GNN) tren 500 mau Train ({epochs} epochs) =====")
    from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device as cpg_to_device
    from torch_geometric.data import Batch, Data
    from src.representations.ast.extract import extract_tree

    def code_to_data(code: str) -> Data:
        tree = extract_tree(code, language="java")
        n = len(tree["label"])
        parents = tree["parent"]
        src, dst = [], []
        for i in range(1, n):
            p = int(parents[i])
            if 0 <= p < n and p != i:
                src.extend([p, i])
                dst.extend([i, p])
        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros(len(src), dtype=torch.long) if src else torch.zeros(0, dtype=torch.long)
        return Data(x=torch.zeros((n, 1), dtype=torch.float), edge_index=edge_index, edge_type=edge_type, node_type=torch.zeros(n, dtype=torch.long), num_nodes=n)

    print("[M4] Build CPG graph cho 500 Train va 500 Test...")
    tr_g1 = [code_to_data(p["func1"]) for p in train_pairs]
    tr_g2 = [code_to_data(p["func2"]) for p in train_pairs]
    tr_y = [p["label"] for p in train_pairs]

    te_g1 = [code_to_data(p["func1"]) for p in test_pairs]
    te_g2 = [code_to_data(p["func2"]) for p in test_pairs]
    te_y = [p["label"] for p in test_pairs]

    model = CPGGNNPairClassifier(hidden=128, num_layers=3, heads=4, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    bs = 32
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        idxs = list(range(len(train_pairs)))
        random.shuffle(idxs)
        for i in range(0, len(idxs), bs):
            chunk = idxs[i : i + bs]
            b1 = cpg_to_device(Batch.from_data_list([tr_g1[j] for j in chunk]), DEVICE)
            b2 = cpg_to_device(Batch.from_data_list([tr_g2[j] for j in chunk]), DEVICE)
            by = torch.tensor([tr_y[j] for j in chunk], dtype=torch.float, device=DEVICE)

            optimizer.zero_grad()
            logits = model(b1, b2)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(chunk)
        train_loss = total_loss / len(train_pairs)

        model.eval()
        with torch.no_grad():
            tr_preds = []
            for i in range(0, len(train_pairs), bs):
                b1 = cpg_to_device(Batch.from_data_list(tr_g1[i : i + bs]), DEVICE)
                b2 = cpg_to_device(Batch.from_data_list(tr_g2[i : i + bs]), DEVICE)
                probs = torch.sigmoid(model(b1, b2)).cpu().tolist()
                tr_preds.extend([1 if p >= 0.5 else 0 for p in probs])
            tr_m = compute_metrics(tr_y, tr_preds)

            te_preds, te_losses = [], []
            for i in range(0, len(test_pairs), bs):
                b1 = cpg_to_device(Batch.from_data_list(te_g1[i : i + bs]), DEVICE)
                b2 = cpg_to_device(Batch.from_data_list(te_g2[i : i + bs]), DEVICE)
                by = torch.tensor(te_y[i : i + bs], dtype=torch.float, device=DEVICE)
                logits = model(b1, b2)
                te_losses.append(criterion(logits, by).item() * len(by))
                probs = torch.sigmoid(logits).cpu().tolist()
                te_preds.extend([1 if p >= 0.5 else 0 for p in probs])
            test_loss = sum(te_losses) / len(test_pairs)
            te_m = compute_metrics(te_y, te_preds)

        print(f"[M4] Epoch {ep}: Train Loss={train_loss:.4f}, Train Acc={tr_m['acc']:.2f}% | Test Loss={test_loss:.4f}, Test Acc={te_m['acc']:.2f}%, Test F1={te_m['f1']:.2f}%")
        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "test_loss": round(test_loss, 4),
            "train_acc": round(tr_m["acc"], 2),
            "test_acc": round(te_m["acc"], 2),
            "test_f1": round(te_m["f1"], 2),
            "test_prec": round(te_m["prec"], 2),
            "test_rec": round(te_m["rec"], 2),
        })

    return history, history[-1]


def main():
    print(f"Device: {DEVICE}")
    train_pairs = load_pairs(TRAIN_FILE)
    test_pairs = load_pairs(TEST_FILE)
    print(f"Loaded {len(train_pairs)} Train pairs và {len(test_pairs)} Test pairs")

    hist_m1, final_m1 = train_transformer_head("microsoft/codebert-base", "M1 (CodeBERT)", train_pairs, test_pairs, epochs=3)
    hist_m2, final_m2 = train_transformer_head("microsoft/graphcodebert-base", "M2 (GraphCodeBERT)", train_pairs, test_pairs, epochs=3)
    hist_m3, final_m3 = train_astnn(train_pairs, test_pairs, epochs=3)
    hist_m4, final_m4 = train_cpg(train_pairs, test_pairs, epochs=5)

    all_history = {
        "M1": hist_m1,
        "M2": hist_m2,
        "M3": hist_m3,
        "M4": hist_m4,
        "final_metrics": {
            "M1": final_m1,
            "M2": final_m2,
            "M3": final_m3,
            "M4": final_m4,
        }
    }
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=2)

    # ==========================================
    # 4. VE BIEU DO SO SANH 4 PHUONG PHAP (500 mau)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    methods = [("M1", "M1: CodeBERT", hist_m1),
               ("M2", "M2: GraphCodeBERT", hist_m2),
               ("M3", "M3: ASTNN", hist_m3),
               ("M4", "M4: CPG-GNN", hist_m4)]

    for idx, (m_id, m_name, h) in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        eps = [x["epoch"] for x in h]
        tr_l = [x["train_loss"] for x in h]
        te_l = [x["test_loss"] for x in h]
        tr_a = [x["train_acc"] for x in h]
        te_a = [x["test_acc"] for x in h]

        ax2 = ax.twinx()
        l1 = ax.plot(eps, tr_l, "o-", color="#1d4ed8", linewidth=2.2, label="Train Loss")
        l2 = ax.plot(eps, te_l, "s--", color="#dc2626", linewidth=2.2, label="Test Loss")
        l3 = ax2.plot(eps, tr_a, "^-", color="#15803d", linewidth=2.0, label="Train Acc (%)")
        l4 = ax2.plot(eps, te_a, "v--", color="#d97706", linewidth=2.0, label="Test Acc (%)")

        ax.set_title(f"{m_name} (Train trên 500 mẫu)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=10, fontweight="bold", color="#1e3a8a")
        ax2.set_ylabel("Accuracy (%)", fontsize=10, fontweight="bold", color="#15803d")
        ax.set_xticks(eps)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Merge legend
        lines = l1 + l2 + l3 + l4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="center right", fontsize=9, frameon=True, facecolor="#ffffff")

    plt.suptitle("ĐƯỜNG CONG LOSS VÀ ACCURACY CỦA 4 PHƯƠNG PHÁP HUẤN LUYỆN TRÊN 500 MẪU TRAIN", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plot_file = FIG_DIR / "comparison_4models_500train.png"
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()

    import shutil
    shutil.copyfile(plot_file, ROOT / "results" / "comparison_4models_500train.png")
    shutil.copyfile(plot_file, Path(r"C:\Users\Kichirou\.gemini\antigravity\brain\99e280e0-b1da-41bb-8efd-bbca4a2661eb\comparison_4models_500train.png"))
    print(f"Bieu do da luu vao: {plot_file}")


if __name__ == "__main__":
    main()
