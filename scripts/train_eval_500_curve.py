#!/usr/bin/env python3
"""Huan luyen M1 (CodeBERT) tren 500 mau Train va danh gia qua cac Epoch tren 500 mau Test.
Ve bieu do Train Loss vs Test Loss va Train Acc vs Test Acc qua cac epoch de kiem tra Overfitting.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.transformer_pair import TransformerPairClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_FILE = ROOT / "results" / "bcb_500_train_pairs.jsonl"
TEST_5K_FILE = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def get_500_test_pairs():
    pairs = []
    with open(TEST_5K_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    
    pos = [p for p in pairs if p["label"] == 1]
    neg = [p for p in pairs if p["label"] == 0]
    rng = random.Random(42)
    rng.shuffle(pos)
    rng.shuffle(neg)
    sub = pos[:250] + neg[:250]
    rng.shuffle(sub)
    return sub


def get_500_train_pairs():
    pairs = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def make_collate_fn(tokenizer, max_length=256):
    def collate_fn(batch):
        c1 = [item["func1"] for item in batch]
        c2 = [item["func2"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
        b1 = tokenizer(c1, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        b2 = tokenizer(c2, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return b1, b2, labels
    return collate_fn


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for b1, b2, labels in loader:
            b1 = {k: v.to(DEVICE) for k, v in b1.items()}
            b2 = {k: v.to(DEVICE) for k, v in b2.items()}
            labels = labels.to(DEVICE)
            logits = model(b1, b2)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
    return total_loss / total, (correct / total) * 100.0


def main():
    print(f"Using device: {DEVICE}")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    train_pairs = get_500_train_pairs()
    test_pairs = get_500_test_pairs()
    print(f"Train samples: {len(train_pairs)} (250 pos : 250 neg)")
    print(f"Test samples:  {len(test_pairs)} (250 pos : 250 neg)")

    hf_id = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    collate = make_collate_fn(tokenizer, max_length=256)

    train_loader = DataLoader(PairDataset(train_pairs), batch_size=16, shuffle=True, collate_fn=collate)
    train_eval_loader = DataLoader(PairDataset(train_pairs), batch_size=32, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_pairs), batch_size=32, shuffle=False, collate_fn=collate)

    model = TransformerPairClassifier(hf_id=hf_id, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 8
    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # Epoch 0 (chua hoc)
    tr_l0, tr_a0 = evaluate(model, train_eval_loader, criterion)
    te_l0, te_a0 = evaluate(model, test_loader, criterion)
    history["epoch"].append(0)
    history["train_loss"].append(tr_l0)
    history["test_loss"].append(te_l0)
    history["train_acc"].append(tr_a0)
    history["test_acc"].append(te_a0)
    print(f"Epoch 0: Train Loss={tr_l0:.4f}, Train Acc={tr_a0:.2f}% | Test Loss={te_l0:.4f}, Test Acc={te_a0:.2f}%")

    for epoch in range(1, num_epochs + 1):
        model.train()
        for b1, b2, labels in train_loader:
            b1 = {k: v.to(DEVICE) for k, v in b1.items()}
            b2 = {k: v.to(DEVICE) for k, v in b2.items()}
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(b1, b2)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        tr_loss, tr_acc = evaluate(model, train_eval_loader, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        print(f"Epoch {epoch:2d}: Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.2f}% | Test Loss={te_loss:.4f}, Test Acc={te_acc:.2f}%")

    # Lưu lịch sử log
    history_file = ROOT / "results" / "learning_curve_500_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {history_file}")

    # Vẽ biểu đồ
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    epochs = history["epoch"]

    # Biểu đồ Loss
    ax1.plot(epochs, history["train_loss"], "o-", color="#1e40af", linewidth=2.2, markersize=6, label="Train Loss (500 mẫu)")
    ax1.plot(epochs, history["test_loss"], "s--", color="#dc2626", linewidth=2.2, markersize=6, label="Test Loss (500 mẫu)")
    ax1.set_title("Đường cong Hàm Mất Mát (Loss Curves)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Binary Cross Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # Biểu đồ Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", color="#16a34a", linewidth=2.2, markersize=6, label="Train Accuracy (500 mẫu)")
    ax2.plot(epochs, history["test_acc"], "s--", color="#ea580c", linewidth=2.2, markersize=6, label="Test Accuracy (500 mẫu)")
    ax2.set_title("Độ Chính Xác (Accuracy Curves)", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.set_ylim(45, 102)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10.5, loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    plt.suptitle("ĐÁNH GIÁ HIỆN TƯỢNG OVERFITTING QUA CÁC EPOCH (CodeBERT M1)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_file = OUTPUT_DIR / "overfit_learning_curves.png"
    plt.savefig(plot_file, bbox_inches="tight")
    plt.close()
    print(f"Chart saved successfully to: {plot_file}")

    # Copy ảnh sang results để người dùng dễ xem
    res_img = ROOT / "results" / "overfit_learning_curves.png"
    import shutil
    shutil.copyfile(plot_file, res_img)
    print(f"Copied to: {res_img}")


if __name__ == "__main__":
    main()
