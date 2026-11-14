"""Sinh 4 biểu đồ trực quan hóa tiến trình huấn luyện chi tiết cho 4 họ mô hình:
  1. reports/figures/training_curves_m1.png (CodeBERT)
  2. reports/figures/training_curves_m2.png (GraphCodeBERT)
  3. reports/figures/training_curves_m3.png (ASTNN)
  4. reports/figures/training_curves_m4.png (CPG-GNN)
Mỗi biểu đồ gồm 2 đồ thị con (Val Loss và Val F1 qua từng Epoch) đối chiếu giữa Baseline và các kịch bản Tinh chỉnh.
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


def plot_m1_curves():
    hist_base = load_history("results/exp_bcb_002_512_v2/training_summary.json")[:8]
    hist_lr = load_history("results/exp_m1_lr/training_summary.json")
    hist_bs = load_history("results/exp_m1_bs/training_summary.json")
    hist_wd = load_history("results/exp_m1_wd/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Val Loss
    if hist_base:
        ax1.plot([h["epoch"] for h in hist_base], [h["val_loss"] for h in hist_base], "o--", color="#7f7f7f", label="M1 Baseline (LR 2e-5, BS 32)", alpha=0.7, linewidth=1.5)
    if hist_bs:
        ax1.plot([h["epoch"] for h in hist_bs], [h["val_loss"] for h in hist_bs], "s-", color="#1f77b4", label="EXP-M1-BS (Eff BS 64)", linewidth=2.0)
    if hist_wd:
        ax1.plot([h["epoch"] for h in hist_wd], [h["val_loss"] for h in hist_wd], "^-", color="#ff7f0e", label="EXP-M1-WD (Weight Decay 0.1)", linewidth=2.0)
    if hist_lr:
        ax1.plot([h["epoch"] for h in hist_lr], [h["val_loss"] for h in hist_lr], "D-", color="#d62728", label="EXP-M1-LR (LR 1e-5) [Best]", linewidth=2.4)
        best_ep = 5
        val_loss_at_best = hist_lr[best_ep - 1]["val_loss"]
        ax1.scatter([best_ep], [val_loss_at_best], color="gold", edgecolor="black", s=140, zorder=5, label=f"Best Ep 5 ({val_loss_at_best:.4f})")

    ax1.set_title("(a) M1 (CodeBERT): Validation Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.5, loc="upper right")

    # 2. Val F1-Score
    if hist_base:
        ax2.plot([h["epoch"] for h in hist_base], [h["val_f1"] * 100 for h in hist_base], "o--", color="#7f7f7f", label="M1 Baseline (512 v2)", alpha=0.7, linewidth=1.5)
    if hist_bs:
        ax2.plot([h["epoch"] for h in hist_bs], [h["val_f1"] * 100 for h in hist_bs], "s-", color="#1f77b4", label="EXP-M1-BS", linewidth=2.0)
    if hist_wd:
        ax2.plot([h["epoch"] for h in hist_wd], [h["val_f1"] * 100 for h in hist_wd], "^-", color="#ff7f0e", label="EXP-M1-WD", linewidth=2.0)
    if hist_lr:
        ax2.plot([h["epoch"] for h in hist_lr], [h["val_f1"] * 100 for h in hist_lr], "D-", color="#2ca02c", label="EXP-M1-LR (Peak 97.04%)", linewidth=2.4)
        ax2.scatter([5], [hist_lr[4]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label="Đỉnh F1 Ep 5 (97.04%)")

    ax2.set_title("(b) M1 (CodeBERT): Validation F1-Score qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_m1.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Da sinh: {out}")


def plot_m2_curves():
    hist_base = load_history("results/exp_bcb_003_512_v2/training_summary.json")[:8]
    hist_lr = load_history("results/exp_m2_lr/training_summary.json")
    hist_bs = load_history("results/exp_m2_bs/training_summary.json")
    hist_dr = load_history("results/exp_m2_dr/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Val Loss
    if hist_base:
        ax1.plot([h["epoch"] for h in hist_base], [h["val_loss"] for h in hist_base], "o--", color="#7f7f7f", label="M2 Baseline (LR 2e-5, BS 32)", alpha=0.7, linewidth=1.5)
    if hist_bs:
        ax1.plot([h["epoch"] for h in hist_bs], [h["val_loss"] for h in hist_bs], "s-", color="#1f77b4", label="EXP-M2-BS (Eff BS 64)", linewidth=2.0)
    if hist_lr:
        ax1.plot([h["epoch"] for h in hist_lr], [h["val_loss"] for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M2-LR (LR 1e-5)", linewidth=2.0)
    if hist_dr:
        ax1.plot([h["epoch"] for h in hist_dr], [h["val_loss"] for h in hist_dr], "D-", color="#9467bd", label="EXP-M2-DR (Dropout 0.3) [Kỷ lục]", linewidth=2.4)
        best_ep = 2
        val_loss_at_best = hist_dr[best_ep - 1]["val_loss"]
        ax1.scatter([best_ep], [val_loss_at_best], color="gold", edgecolor="black", s=140, zorder=5, label=f"Best Ep 2 ({val_loss_at_best:.4f})")

    ax1.set_title("(a) M2 (GraphCodeBERT): Validation Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8.5, loc="upper right")

    # 2. Val F1-Score
    if hist_base:
        ax2.plot([h["epoch"] for h in hist_base], [h["val_f1"] * 100 for h in hist_base], "o--", color="#7f7f7f", label="M2 Baseline (512 v2)", alpha=0.7, linewidth=1.5)
    if hist_bs:
        ax2.plot([h["epoch"] for h in hist_bs], [h["val_f1"] * 100 for h in hist_bs], "s-", color="#1f77b4", label="EXP-M2-BS", linewidth=2.0)
    if hist_lr:
        ax2.plot([h["epoch"] for h in hist_lr], [h["val_f1"] * 100 for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M2-LR", linewidth=2.0)
    if hist_dr:
        ax2.plot([h["epoch"] for h in hist_dr], [h["val_f1"] * 100 for h in hist_dr], "D-", color="#2ca02c", label="EXP-M2-DR (Peak 96.95% Val)", linewidth=2.4)
        ax2.scatter([2], [hist_dr[1]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label="Đỉnh F1 Ep 2 (96.95%)")

    ax2.set_title("(b) M2 (GraphCodeBERT): Validation F1-Score qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8.5, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_m2.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Da sinh: {out}")


def plot_m3_curves():
    hist_opt = load_history("results/exp_bcb_004_opt/training_summary.json")
    hist_ep = load_history("results/exp_m3_ep/training_summary.json")
    hist_lr = load_history("results/exp_m3_lr/training_summary.json")
    hist_bs = load_history("results/exp_m3_bs/training_summary.json")
    hist_hd = load_history("results/exp_m3_hd/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Val Loss
    if hist_opt:
        ax1.plot([h["epoch"] for h in hist_opt], [h["val_loss"] for h in hist_opt], "o--", color="#7f7f7f", label="M3 Opt Baseline (8 Ep, LR 3e-4)", alpha=0.8, linewidth=1.6)
    if hist_lr:
        ax1.plot([h["epoch"] for h in hist_lr], [h["val_loss"] for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M3-LR (LR 1e-4)", linewidth=1.8)
    if hist_bs:
        ax1.plot([h["epoch"] for h in hist_bs], [h["val_loss"] for h in hist_bs], "s-", color="#1f77b4", label="EXP-M3-BS (Batch Size 64)", linewidth=1.8)
    if hist_hd:
        ax1.plot([h["epoch"] for h in hist_hd], [h["val_loss"] for h in hist_hd], "v-", color="#9467bd", label="EXP-M3-HD (Hidden Size 300)", linewidth=1.8)
    if hist_ep:
        ax1.plot([h["epoch"] for h in hist_ep], [h["val_loss"] for h in hist_ep], "D-", color="#d62728", label="EXP-M3-EP (Nới rộng 20 Epochs) [Best]", linewidth=2.4)
        ax1.scatter([19], [hist_ep[18]["val_loss"]], color="gold", edgecolor="black", s=140, zorder=5, label=f"Min Loss Ep 19 ({hist_ep[18]['val_loss']:.4f})")

    ax1.set_title("(a) M3 (ASTNN): Validation Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right")

    # 2. Val F1-Score
    if hist_opt:
        ax2.plot([h["epoch"] for h in hist_opt], [h["val_f1"] * 100 for h in hist_opt], "o--", color="#7f7f7f", label="M3 Opt Baseline (8 Ep)", alpha=0.8, linewidth=1.6)
    if hist_lr:
        ax2.plot([h["epoch"] for h in hist_lr], [h["val_f1"] * 100 for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M3-LR", linewidth=1.8)
    if hist_bs:
        ax2.plot([h["epoch"] for h in hist_bs], [h["val_f1"] * 100 for h in hist_bs], "s-", color="#1f77b4", label="EXP-M3-BS", linewidth=1.8)
    if hist_hd:
        ax2.plot([h["epoch"] for h in hist_hd], [h["val_f1"] * 100 for h in hist_hd], "v-", color="#9467bd", label="EXP-M3-HD", linewidth=1.8)
    if hist_ep:
        ax2.plot([h["epoch"] for h in hist_ep], [h["val_f1"] * 100 for h in hist_ep], "D-", color="#2ca02c", label="EXP-M3-EP (Đỉnh 76.22% Val)", linewidth=2.4)
        ax2.scatter([19], [hist_ep[18]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label="Đỉnh F1 Ep 19 (76.22%)")

    ax2.set_title("(b) M3 (ASTNN): Validation F1-Score qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_m3.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Da sinh: {out}")


def plot_m4_curves():
    hist_opt = load_history("results/exp_bcb_005_opt/training_summary.json")
    hist_ep = load_history("results/exp_m4_ep/training_summary.json")
    hist_bs = load_history("results/exp_m4_bs/training_summary.json")
    hist_hd = load_history("results/exp_m4_hd/training_summary.json")
    hist_lr = load_history("results/exp_m4_lr/training_summary.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Val Loss
    if hist_opt:
        ax1.plot([h["epoch"] for h in hist_opt], [h["val_loss"] for h in hist_opt], "o--", color="#7f7f7f", label="M4 Opt Baseline (8 Ep, LR 3e-4)", alpha=0.8, linewidth=1.6)
    if hist_lr:
        ax1.plot([h["epoch"] for h in hist_lr], [h["val_loss"] for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M4-LR (LR 1e-4)", linewidth=1.8)
    if hist_bs:
        ax1.plot([h["epoch"] for h in hist_bs], [h["val_loss"] for h in hist_bs], "s-", color="#1f77b4", label="EXP-M4-BS (Batch Size 64)", linewidth=1.8)
    if hist_hd:
        ax1.plot([h["epoch"] for h in hist_hd], [h["val_loss"] for h in hist_hd], "v-", color="#9467bd", label="EXP-M4-HD (Attention Heads 8)", linewidth=1.8)
    if hist_ep:
        ax1.plot([h["epoch"] for h in hist_ep], [h["val_loss"] for h in hist_ep], "D-", color="#d62728", label="EXP-M4-EP (Nới rộng 20 Epochs) [Best]", linewidth=2.4)
        ax1.scatter([17], [hist_ep[16]["val_loss"]], color="gold", edgecolor="black", s=140, zorder=5, label=f"Min Loss Ep 17 ({hist_ep[16]['val_loss']:.4f})")

    ax1.set_title("(a) M4 (CPG-GNN): Validation Loss qua các Epoch", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="upper right")

    # 2. Val F1-Score
    if hist_opt:
        ax2.plot([h["epoch"] for h in hist_opt], [h["val_f1"] * 100 for h in hist_opt], "o--", color="#7f7f7f", label="M4 Opt Baseline (8 Ep)", alpha=0.8, linewidth=1.6)
    if hist_lr:
        ax2.plot([h["epoch"] for h in hist_lr], [h["val_f1"] * 100 for h in hist_lr], "^-", color="#ff7f0e", label="EXP-M4-LR", linewidth=1.8)
    if hist_bs:
        ax2.plot([h["epoch"] for h in hist_bs], [h["val_f1"] * 100 for h in hist_bs], "s-", color="#1f77b4", label="EXP-M4-BS", linewidth=1.8)
    if hist_hd:
        ax2.plot([h["epoch"] for h in hist_hd], [h["val_f1"] * 100 for h in hist_hd], "v-", color="#9467bd", label="EXP-M4-HD", linewidth=1.8)
    if hist_ep:
        ax2.plot([h["epoch"] for h in hist_ep], [h["val_f1"] * 100 for h in hist_ep], "D-", color="#2ca02c", label="EXP-M4-EP (Đỉnh 80.45% Val)", linewidth=2.4)
        ax2.scatter([17], [hist_ep[16]["val_f1"] * 100], color="gold", edgecolor="black", s=140, zorder=5, label="Đỉnh F1 Ep 17 (80.45%)")

    ax2.set_title("(b) M4 (CPG-GNN): Validation F1-Score qua các Epoch", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Validation F1-Score (%)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "training_curves_m4.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Da sinh: {out}")


if __name__ == "__main__":
    plot_m1_curves()
    plot_m2_curves()
    plot_m3_curves()
    plot_m4_curves()
    print(">>> HOAN TAT VE 4 BIEU DO TIEN TRINH HUAN LUYEN! <<<")
