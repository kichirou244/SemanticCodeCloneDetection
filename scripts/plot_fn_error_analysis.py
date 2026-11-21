"""Vẽ biểu đồ phân tích trực quan các ca lỗi False Negative (FN) cho báo cáo LaTeX."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "reports" / "figures" / "fn_error_analysis.png"

def plot_fn_analysis():
    # Cấu hình font và style học thuật
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=300)

    # -------------------------------------------------------------
    # Subplot 1: Phân bố xác suất của các ca FN (Histogram)
    # -------------------------------------------------------------
    ax1 = axes[0]
    bins_labels = ["[0, 0.05)", "[0.05, 0.15)", "[0.15, 0.30)", "[0.30, 0.45)", "[0.45, 0.52)"]
    counts = [85, 23, 5, 5, 7]
    pcts = [68.0, 18.4, 4.0, 4.0, 5.6]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5bc0de", "#428bca"]

    bars = ax1.bar(bins_labels, counts, color=colors, width=0.62, edgecolor="#333333", linewidth=0.7)
    ax1.set_title("(a) Phân bố xác suất dự đoán $P$ của FN", fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Khoảng xác suất dự đoán $P(y=1)$", fontsize=9.5)
    ax1.set_ylabel("Số lượng mẫu FN (Tổng: 125)", fontsize=9.5)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 2.2, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

    # -------------------------------------------------------------
    # Subplot 2: So sánh đặc trưng TP vs FN vs FP
    # -------------------------------------------------------------
    ax2 = axes[1]
    metrics = ["Tỷ lệ cắt cụt\n(>512 tokens)", "Trùng từ vựng\n(Jaccard %)", "Lệch độ dài\n(Length Ratio)"]
    tp_vals = [32.8, 17.7, 2.55 * 10]    # nhân 10 để cùng thang đo trực quan
    fn_vals = [60.0, 11.9, 3.32 * 10]
    fp_vals = [40.4, 14.5, 2.35 * 10]

    x = np.arange(len(metrics))
    width = 0.25

    rects1 = ax2.bar(x - width, tp_vals, width, label="Đoán đúng (TP)", color="#28a745", edgecolor="#333", alpha=0.9)
    rects2 = ax2.bar(x, fn_vals, width, label="Bỏ sót (FN)", color="#dc3545", edgecolor="#333", alpha=0.9)
    rects3 = ax2.bar(x + width, fp_vals, width, label="Báo động nhầm (FP)", color="#ffc107", edgecolor="#333", alpha=0.9)

    ax2.set_title("(b) Đối chiếu đặc trưng mã nguồn", fontsize=11, fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylabel("Giá trị chỉ số (%)", fontsize=9.5)
    ax2.set_ylim(0, 75)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="upper right")

    # Ghi chú giá trị trên cột
    for rect, real_val in [(rects1[0], "32.8%"), (rects2[0], "60.0%"), (rects3[0], "40.4%"),
                           (rects1[1], "17.7%"), (rects2[1], "11.9%"), (rects3[1], "14.5%"),
                           (rects1[2], "2.55x"), (rects2[2], "3.32x"), (rects3[2], "2.35x")]:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 1.5, real_val, ha="center", va="bottom", fontsize=8, fontweight="semibold")

    # -------------------------------------------------------------
    # Subplot 3: Khả năng giải cứu lỗi của CPG-GNN (Donut Chart)
    # -------------------------------------------------------------
    ax3 = axes[2]
    sizes = [100, 25]
    labels = ["CPG-GNN cứu đúng\n(100 ca - 80.0%)", "Hard FN (Cả hai sai)\n(25 ca - 20.0%)"]
    donut_colors = ["#17a2b8", "#6c757d"]

    wedges, texts, autotexts = ax3.pie(
        sizes, labels=labels, autopct="%1.0f%%", startangle=140,
        colors=donut_colors, pctdistance=0.72,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=8.5, fontweight="semibold")
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax3.set_title("(c) CPG-GNN (M4) giải cứu FN của M2", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Đã tạo biểu đồ trực quan học thuật tại: {OUT_PATH}")

if __name__ == "__main__":
    plot_fn_analysis()
