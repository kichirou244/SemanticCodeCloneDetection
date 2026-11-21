"""Sinh biểu đồ trực quan hóa tiến trình huấn luyện chi tiết cho 2 mô hình tương tác mới:
  1. reports/figures/training_curves_diffmul_m1.png (EXP-M1-DiffMul)
  2. reports/figures/training_curves_diffmul_m2.png (EXP-M2-DiffMul)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path_str: str) -> list[dict]:
    p = Path(path_str)
    if not p.exists():
        print(f"Khong tim thay: {path_str}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("history", [])


def plot_m1_diffmul():
    hist_diff = load_history("results/exp_m1_diffmul/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Loss
    if hist_diff:
        ax1.plot([h["epoch"] for h in hist_diff], [h["train_loss"] for h in hist_diff], "s-", color="#1f77b4", label="Train Loss", linewidth=2.2)
        ax1.plot([h["epoch"] for h in hist_diff], [h["val_loss"] for h in hist_diff], "D-", color="#d62728", label="Val Loss", linewidth=2.5)
        best_l_idx = min(range(len(hist_diff)), key=lambda i: hist_diff[i]["val_loss"])
        ax1.scatter([hist_diff[best_l_idx]["epoch"]], [hist_diff[best_l_idx]["val_loss"]], color="gold", edgecolor="black", s=140, zorder=5, label=f"Best Val Loss Ep {hist_diff[best_l_idx]['epoch']} ({hist_diff[best_l_idx]['val_loss']:.4f})")

    ax1.set_title("(a) EXP-M1-DiffMul: Tiến trình Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="upper right")

    # 2. Validation F1-Score
    if hist_diff:
        ax2.plot([h["epoch"] for h in hist_diff], [h["val_f1"] * 100 for h in hist_diff], "D-", color="#2ca02c", label="Val F1-Score", linewidth=2.5)
        best_f1_idx = max(range(len(hist_diff)), key=lambda i: hist_diff[i]["val_f1"])
        ax2.scatter([hist_diff[best_f1_idx]["epoch"]], [hist_diff[best_f1_idx]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label=f"Đỉnh F1 Ep {hist_diff[best_f1_idx]['epoch']} ({hist_diff[best_f1_idx]['val_f1']*100:.2f}%)")

    ax2.set_title("(b) EXP-M1-DiffMul: Validation F1 qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_diffmul_m1.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Đã sinh thành công: {out}")


def plot_m2_diffmul():
    hist_diff = load_history("results/exp_m2_diffmul/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Loss
    if hist_diff:
        ax1.plot([h["epoch"] for h in hist_diff], [h["train_loss"] for h in hist_diff], "s-", color="#1f77b4", label="Train Loss", linewidth=2.2)
        ax1.plot([h["epoch"] for h in hist_diff], [h["val_loss"] for h in hist_diff], "D-", color="#d62728", label="Val Loss", linewidth=2.5)
        best_l_idx = min(range(len(hist_diff)), key=lambda i: hist_diff[i]["val_loss"])
        ax1.scatter([hist_diff[best_l_idx]["epoch"]], [hist_diff[best_l_idx]["val_loss"]], color="gold", edgecolor="black", s=140, zorder=5, label=f"Best Val Loss Ep {hist_diff[best_l_idx]['epoch']} ({hist_diff[best_l_idx]['val_loss']:.4f})")

    ax1.set_title("(a) EXP-M2-DiffMul: Tiến trình Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9, loc="upper right")

    # 2. Validation F1-Score
    if hist_diff:
        ax2.plot([h["epoch"] for h in hist_diff], [h["val_f1"] * 100 for h in hist_diff], "D-", color="#2ca02c", label="Val F1-Score", linewidth=2.5)
        best_f1_idx = max(range(len(hist_diff)), key=lambda i: hist_diff[i]["val_f1"])
        ax2.scatter([hist_diff[best_f1_idx]["epoch"]], [hist_diff[best_f1_idx]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label=f"Đỉnh F1 Ep {hist_diff[best_f1_idx]['epoch']} ({hist_diff[best_f1_idx]['val_f1']*100:.2f}%)")

    ax2.set_title("(b) EXP-M2-DiffMul: Validation F1 qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_diffmul_m2.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Đã sinh thành công: {out}")


if __name__ == "__main__":
    plot_m1_diffmul()
    plot_m2_diffmul()
