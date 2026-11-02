"""Vẽ biểu đồ so sánh tiến trình huấn luyện (Loss & Accuracy) giữa 384 tokens và 512 tokens cho M1 (CodeBERT) và M2 (GraphCodeBERT)."""

import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Nạp dữ liệu history
h_m1_384 = json.load(open("results/exp_bcb_002_384/training_summary.json"))["history"]
h_m1_512 = json.load(open("results/exp_bcb_002_512/training_summary.json"))["history"]
h_m2_384 = json.load(open("results/exp_bcb_003_384/training_summary.json"))["history"]
h_m2_512 = json.load(open("results/exp_bcb_003_512/training_summary.json"))["history"]

epochs = [1, 2, 3]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)

# Colors & styles
c_384 = "#1f77b4"  # Blue
c_512 = "#d62728"  # Red

# 1. Top-Left: M1 Train & Val Loss
ax = axes[0, 0]
ax.plot(epochs, [h["train_loss"] for h in h_m1_384], "o--", color=c_384, label="M1-384 Train Loss", alpha=0.8)
ax.plot(epochs, [h["val_loss"] for h in h_m1_384], "s-", color=c_384, label="M1-384 Val Loss", linewidth=2)
ax.plot(epochs, [h["train_loss"] for h in h_m1_512], "o--", color=c_512, label="M1-512 Train Loss", alpha=0.8)
ax.plot(epochs, [h["val_loss"] for h in h_m1_512], "s-", color=c_512, label="M1-512 Val Loss", linewidth=2)
ax.set_title("(a) M1 (CodeBERT): Train Loss & Val Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax.set_xticks(epochs)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 2. Top-Right: M1 Val Accuracy
ax = axes[0, 1]
ax.plot(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m1_384], "o-", color=c_384, label="M1-384 Tokens", linewidth=2.2, markersize=7)
ax.plot(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m1_512], "s-", color=c_512, label="M1-512 Tokens", linewidth=2.2, markersize=7)
for ep, v in zip(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m1_384]):
    ax.annotate(f"{v:.2f}%", (ep, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=c_384)
for ep, v in zip(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m1_512]):
    ax.annotate(f"{v:.2f}%", (ep, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=c_512, fontweight="bold")
ax.set_title("(b) M1 (CodeBERT): Validation Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(epochs)
ax.set_ylim(95.5, 98.5)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 3. Bottom-Left: M2 Train & Val Loss
ax = axes[1, 0]
ax.plot(epochs, [h["train_loss"] for h in h_m2_384], "o--", color=c_384, label="M2-384 Train Loss", alpha=0.8)
ax.plot(epochs, [h["val_loss"] for h in h_m2_384], "s-", color=c_384, label="M2-384 Val Loss", linewidth=2)
ax.plot(epochs, [h["train_loss"] for h in h_m2_512], "o--", color=c_512, label="M2-512 Train Loss", alpha=0.8)
ax.plot(epochs, [h["val_loss"] for h in h_m2_512], "s-", color=c_512, label="M2-512 Val Loss", linewidth=2)
ax.set_title("(c) M2 (GraphCodeBERT): Train Loss & Val Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax.set_xticks(epochs)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 4. Bottom-Right: M2 Val Accuracy
ax = axes[1, 1]
ax.plot(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m2_384], "o-", color=c_384, label="M2-384 Tokens", linewidth=2.2, markersize=7)
ax.plot(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m2_512], "s-", color=c_512, label="M2-512 Tokens", linewidth=2.2, markersize=7)
for ep, v in zip(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m2_384]):
    ax.annotate(f"{v:.2f}%", (ep, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=c_384)
for ep, v in zip(epochs, [h["metrics"]["accuracy"] * 100 for h in h_m2_512]):
    ax.annotate(f"{v:.2f}%", (ep, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=c_512, fontweight="bold")
ax.set_title("(d) M2 (GraphCodeBERT): Validation Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(epochs)
ax.set_ylim(95.5, 98.5)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
out_path = Path("reports/figures/compare_tokens_384_vs_512.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Đã xuất biểu đồ: {out_path}")
