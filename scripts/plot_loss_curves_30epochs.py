"""Vẽ biểu đồ tiến trình huấn luyện (Train Loss vs Val Loss & Val F1/Accuracy) qua các Epoch (hỗ trợ tối đa 30+ epochs)

cho M1 (CodeBERT 512) và M2 (GraphCodeBERT 512).
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def plot_curves(m1_path: Path, m2_path: Path, out_path: Path):
    h_m1 = []
    h_m2 = []
    
    if m1_path.exists():
        with open(m1_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            h_m1 = data.get("history", [])
            print(f"Nạp M1 history ({len(h_m1)} epochs) từ {m1_path}")
    else:
        print(f"Cảnh báo: Không tìm thấy file M1 tại {m1_path}")

    if m2_path.exists():
        with open(m2_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            h_m2 = data.get("history", [])
            print(f"Nạp M2 history ({len(h_m2)} epochs) từ {m2_path}")
    else:
        print(f"Cảnh báo: Không tìm thấy file M2 tại {m2_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=300)

    c_train = "#1f77b4"  # Blue
    c_val = "#d62728"    # Red
    c_f1 = "#2ca02c"     # Green
    c_acc = "#9467bd"    # Purple

    # 1. Subplot (a): M1 Train & Val Loss
    ax = axes[0, 0]
    if h_m1:
        eps1 = [h["epoch"] for h in h_m1]
        t_loss1 = [h["train_loss"] for h in h_m1]
        v_loss1 = [h["val_loss"] for h in h_m1]
        ax.plot(eps1, t_loss1, "o--", color=c_train, label="M1 Train Loss", alpha=0.85, linewidth=1.8)
        ax.plot(eps1, v_loss1, "s-", color=c_val, label="M1 Val Loss", linewidth=2.2)
        # Đánh dấu min val loss
        min_idx = v_loss1.index(min(v_loss1))
        ax.scatter([eps1[min_idx]], [v_loss1[min_idx]], color="gold", edgecolor="black", s=130, zorder=5, label=f"Min Val Loss ({v_loss1[min_idx]:.4f})")
    ax.set_title("(a) M1 - CodeBERT (512 tokens): Train vs Val Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")

    # 2. Subplot (b): M1 Val Metrics
    ax = axes[0, 1]
    if h_m1:
        eps1 = [h["epoch"] for h in h_m1]
        f1_1 = [h.get("val_f1", h.get("metrics", {}).get("f1", 0.0)) * 100 for h in h_m1]
        acc_1 = [h.get("metrics", {}).get("accuracy", 0.0) * 100 for h in h_m1]
        ax.plot(eps1, f1_1, "^-", color=c_f1, label="Val F1-Score (%)", linewidth=2)
        ax.plot(eps1, acc_1, "d-", color=c_acc, label="Val Accuracy (%)", linewidth=2)
    ax.set_title("(b) M1 - CodeBERT (512 tokens): Val F1 & Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")

    # 3. Subplot (c): M2 Train & Val Loss
    ax = axes[1, 0]
    if h_m2:
        eps2 = [h["epoch"] for h in h_m2]
        t_loss2 = [h["train_loss"] for h in h_m2]
        v_loss2 = [h["val_loss"] for h in h_m2]
        ax.plot(eps2, t_loss2, "o--", color=c_train, label="M2 Train Loss", alpha=0.85, linewidth=1.8)
        ax.plot(eps2, v_loss2, "s-", color=c_val, label="M2 Val Loss", linewidth=2.2)
        min_idx2 = v_loss2.index(min(v_loss2))
        ax.scatter([eps2[min_idx2]], [v_loss2[min_idx2]], color="gold", edgecolor="black", s=130, zorder=5, label=f"Min Val Loss ({v_loss2[min_idx2]:.4f})")
    ax.set_title("(c) M2 - GraphCodeBERT (512 tokens): Train vs Val Loss", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")

    # 4. Subplot (d): M2 Val Metrics
    ax = axes[1, 1]
    if h_m2:
        eps2 = [h["epoch"] for h in h_m2]
        f1_2 = [h.get("val_f1", h.get("metrics", {}).get("f1", 0.0)) * 100 for h in h_m2]
        acc_2 = [h.get("metrics", {}).get("accuracy", 0.0) * 100 for h in h_m2]
        ax.plot(eps2, f1_2, "^-", color=c_f1, label="Val F1-Score (%)", linewidth=2)
        ax.plot(eps2, acc_2, "d-", color=c_acc, label="Val Accuracy (%)", linewidth=2)
    ax.set_title("(d) M2 - GraphCodeBERT (512 tokens): Val F1 & Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Đã xuất biểu đồ theo dõi Loss/Metrics vào: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Vẽ loss curve cho M1 & M2")
    parser.add_argument("--m1-summary", type=Path, default=Path("results/exp_bcb_002_512/training_summary.json"))
    parser.add_argument("--m2-summary", type=Path, default=Path("results/exp_bcb_003_512/training_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/figures/loss_curves_30epochs_m1_m2.png"))
    args = parser.parse_args()

    plot_curves(args.m1_summary, args.m2_summary, args.out)


if __name__ == "__main__":
    main()
