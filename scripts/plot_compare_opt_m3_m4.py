"""Vẽ biểu đồ so sánh tiến trình huấn luyện (Loss & Accuracy) giữa Bản Ban đầu và Bản Tối ưu (Opt) cho M3 (ASTNN) và M4 (CPG-GNN)."""

import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Nạp dữ liệu history
h_m3_raw = json.load(open("results/exp_bcb_004/training_summary.json"))["history"]
h_m3_opt = json.load(open("results/exp_bcb_004_opt/training_summary.json"))["history"]
h_m4_raw = json.load(open("results/exp_bcb_005/training_summary.json"))["history"]
h_m4_opt = json.load(open("results/exp_bcb_005_opt/training_summary.json"))["history"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)

c_raw = "#7f7f7f"  # Gray for baseline/raw
c_opt = "#2ca02c"  # Green for optimized

# 1. Top-Left: M3 Train & Val Loss
ax = axes[0, 0]
ep_raw = list(range(1, len(h_m3_raw) + 1))
ep_opt = list(range(1, len(h_m3_opt) + 1))

ax.plot(ep_raw, [h["train_loss"] for h in h_m3_raw], "o--", color=c_raw, label="M3 Ban đầu (Train Loss)", alpha=0.7)
ax.plot(ep_raw, [h["val_loss"] for h in h_m3_raw], "s-", color=c_raw, label="M3 Ban đầu (Val Loss)", linewidth=2, alpha=0.8)
ax.plot(ep_opt, [h["train_loss"] for h in h_m3_opt], "o--", color=c_opt, label="M3 Opt (Train Loss)", alpha=0.8)
ax.plot(ep_opt, [h["val_loss"] for h in h_m3_opt], "s-", color=c_opt, label="M3 Opt (Val Loss)", linewidth=2)

ax.set_title("(a) M3 (ASTNN): Train Loss & Val Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax.set_xticks(ep_opt)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 2. Top-Right: M3 Val Accuracy
ax = axes[0, 1]
ax.plot(ep_raw, [h["metrics"]["accuracy"] * 100 for h in h_m3_raw], "o-", color=c_raw, label="M3 Ban đầu", linewidth=2.2, markersize=7)
ax.plot(ep_opt, [h["metrics"]["accuracy"] * 100 for h in h_m3_opt], "s-", color=c_opt, label="M3 Tối ưu (Opt)", linewidth=2.2, markersize=7)

for ep, v in zip(ep_raw, [h["metrics"]["accuracy"] * 100 for h in h_m3_raw]):
    ax.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=c_raw)
for ep, v in zip(ep_opt, [h["metrics"]["accuracy"] * 100 for h in h_m3_opt]):
    ax.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=c_opt, fontweight="bold")

ax.set_title("(b) M3 (ASTNN): Validation Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(ep_opt)
ax.set_ylim(10, 85)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 3. Bottom-Left: M4 Train & Val Loss
ax = axes[1, 0]
ep4_raw = list(range(1, len(h_m4_raw) + 1))
ep4_opt = list(range(1, len(h_m4_opt) + 1))

ax.plot(ep4_raw, [h["train_loss"] for h in h_m4_raw], "o--", color=c_raw, label="M4 Ban đầu (Train Loss)", alpha=0.7)
ax.plot(ep4_raw, [h["val_loss"] for h in h_m4_raw], "s-", color=c_raw, label="M4 Ban đầu (Val Loss)", linewidth=2, alpha=0.8)
ax.plot(ep4_opt, [h["train_loss"] for h in h_m4_opt], "o--", color=c_opt, label="M4 Opt (Train Loss)", alpha=0.8)
ax.plot(ep4_opt, [h["val_loss"] for h in h_m4_opt], "s-", color=c_opt, label="M4 Opt (Val Loss)", linewidth=2)

ax.set_title("(c) M4 (CPG-GNN): Train Loss & Val Loss", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
ax.set_xticks(ep4_opt)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

# 4. Bottom-Right: M4 Val Accuracy
ax = axes[1, 1]
ax.plot(ep4_raw, [h["metrics"]["accuracy"] * 100 for h in h_m4_raw], "o-", color=c_raw, label="M4 Ban đầu", linewidth=2.2, markersize=7)
ax.plot(ep4_opt, [h["metrics"]["accuracy"] * 100 for h in h_m4_opt], "s-", color=c_opt, label="M4 Tối ưu (Opt)", linewidth=2.2, markersize=7)

for ep, v in zip(ep4_raw, [h["metrics"]["accuracy"] * 100 for h in h_m4_raw]):
    ax.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=c_raw)
for ep, v in zip(ep4_opt, [h["metrics"]["accuracy"] * 100 for h in h_m4_opt]):
    ax.annotate(f"{v:.1f}%", (ep, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5, color=c_opt, fontweight="bold")

ax.set_title("(d) M4 (CPG-GNN): Validation Accuracy", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=10)
ax.set_ylabel("Accuracy (%)", fontsize=10)
ax.set_xticks(ep4_opt)
ax.set_ylim(10, 90)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=9, loc="lower left")

plt.tight_layout()
out_path = Path("reports/figures/compare_opt_m3_m4.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)
plt.close()
print(f"Đã xuất biểu đồ: {out_path}")
