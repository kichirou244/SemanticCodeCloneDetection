#!/usr/bin/env python3
"""Ve bieu do Learning Curves chuan cho M1 (CodeBERT)
dua tren dung so epoch thuc te khi huan luyen (5 epoch).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = ROOT / "results" / "bcb_500_train_pairs.jsonl"
TEST_5K_FILE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MLPPairHead(nn.Module):
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


def load_pairs(filepath, n_pos=250, n_neg=250):
    pairs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sub = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(sub)
    return sub


def extract_features(pairs, backbone, tokenizer, bs=64):
    c1 = [p["func1"] for p in pairs]
    c2 = [p["func2"] for p in pairs]
    labels = torch.tensor([p["label"] for p in pairs], dtype=torch.float)

    h1_list, h2_list = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            b1 = tokenizer(c1[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            b2 = tokenizer(c2[i : i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
            out1 = backbone(**b1).last_hidden_state[:, 0, :].cpu()
            out2 = backbone(**b2).last_hidden_state[:, 0, :].cpu()
            h1_list.append(out1)
            h2_list.append(out2)
    return torch.cat(h1_list, dim=0), torch.cat(h2_list, dim=0), labels


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_pairs = load_pairs(TRAIN_FILE, 250, 250)
    test_pairs = load_pairs(TEST_5K_FILE, 250, 250)

    hf_id = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    backbone = AutoModel.from_pretrained(hf_id).to(DEVICE)
    backbone.eval()

    tr_h1, tr_h2, tr_y = extract_features(train_pairs, backbone, tokenizer)
    te_h1, te_h2, te_y = extract_features(test_pairs, backbone, tokenizer)

    del backbone
    torch.cuda.empty_cache()

    train_ds = TensorDataset(tr_h1, tr_h2, tr_y)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    head = MLPPairHead().to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 5
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    def evaluate_tensors(h1, h2, y):
        head.eval()
        with torch.no_grad():
            h1_d, h2_d, y_d = h1.to(DEVICE), h2.to(DEVICE), y.to(DEVICE)
            logits = head(h1_d, h2_d)
            loss = criterion(logits, y_d).item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = (preds == y_d).float().mean().item() * 100.0
        return loss, acc

    # Epoch 0
    l_tr0, a_tr0 = evaluate_tensors(tr_h1, tr_h2, tr_y)
    l_te0, a_te0 = evaluate_tensors(te_h1, te_h2, te_y)
    history["epoch"].append(0)
    history["train_loss"].append(round(l_tr0, 4))
    history["test_loss"].append(round(l_te0, 4))
    history["train_acc"].append(round(a_tr0, 2))
    history["test_acc"].append(round(a_te0, 2))

    for epoch in range(1, epochs + 1):
        head.train()
        for b_h1, b_h2, b_y in train_loader:
            b_h1, b_h2, b_y = b_h1.to(DEVICE), b_h2.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            logits = head(b_h1, b_h2)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

        l_tr, a_tr = evaluate_tensors(tr_h1, tr_h2, tr_y)
        l_te, a_te = evaluate_tensors(te_h1, te_h2, te_y)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(l_tr, 4))
        history["test_loss"].append(round(l_te, 4))
        history["train_acc"].append(round(a_tr, 2))
        history["test_acc"].append(round(a_te, 2))
        print(f"Epoch {epoch}: Train Loss={l_tr:.4f}, Train Acc={a_tr:.2f}% | Test Loss={l_te:.4f}, Test Acc={a_te:.2f}%")

    with open(ROOT / "results" / "learning_curve_500_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Vẽ biểu đồ chuẩn 5 epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ep_list = history["epoch"]

    # 1. Loss
    ax1.plot(ep_list, history["train_loss"], "o-", color="#1d4ed8", linewidth=2.4, markersize=7, label="Train Loss (500 mẫu)")
    ax1.plot(ep_list, history["test_loss"], "s--", color="#dc2626", linewidth=2.4, markersize=7, label="Test Loss (500 mẫu)")
    ax1.set_title("Hàm mất mát qua các Epoch (Loss)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(ep_list)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # 2. Accuracy
    ax2.plot(ep_list, history["train_acc"], "o-", color="#15803d", linewidth=2.4, markersize=7, label="Train Accuracy (500 mẫu)")
    ax2.plot(ep_list, history["test_acc"], "s--", color="#ea580c", linewidth=2.4, markersize=7, label="Test Accuracy (500 mẫu)")
    ax2.set_title("Độ chính xác qua các Epoch (Accuracy)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(ep_list)
    ax2.set_ylim(45, 95)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    plt.suptitle("ĐƯỜNG CONG HỌC TẬP (LEARNING CURVES) TRÊN 500 MẪU TRAIN VÀ 500 MẪU TEST (5 EPOCHS)", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_fig = OUTPUT_DIR / "overfit_learning_curves.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    import shutil
    shutil.copyfile(out_fig, ROOT / "results" / "overfit_learning_curves.png")
    shutil.copyfile(out_fig, Path(r"C:\Users\Kichirou\.gemini\antigravity\brain\99e280e0-b1da-41bb-8efd-bbca4a2661eb\overfit_learning_curves.png"))
    print("Xong!")


if __name__ == "__main__":
    main()
