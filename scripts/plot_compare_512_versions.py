"""Vẽ biểu đồ đối chiếu hiệu năng Test giữa các phiên bản V1 và V2 của M1 và M2 (512 tokens)."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

models = {
    "M1 (CodeBERT 512)": [
        ("v1 (Baseline 3ep)", "results/exp_bcb_002_512/test_metrics.json"),
        ("v2 (Extended 12ep)", "results/exp_bcb_002_512_v2/test_metrics.json"),
    ],
    "M2 (GraphCodeBERT 512)": [
        ("v1 (Baseline 3ep)", "results/exp_bcb_003_512/test_metrics.json"),
        ("v2 (Extended 13ep)", "results/exp_bcb_003_512_v2/test_metrics.json"),
    ],
}

metrics_keys = ["f1", "accuracy", "precision", "recall", "roc_auc"]
metrics_labels = ["F1-Score", "Accuracy", "Precision", "Recall", "ROC-AUC"]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

colors = ["#1f77b4", "#d95f02"]
x = np.arange(len(metrics_keys))
bar_width = 0.34

for ax_idx, (model_title, versions) in enumerate(models.items()):
    ax = axes[ax_idx]
    
    for i, (ver_name, fpath) in enumerate(versions):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vals = [data[k] * 100 for k in metrics_keys]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, vals, width=bar_width, label=ver_name, color=colors[i], alpha=0.88, edgecolor="black", linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.2, rotation=0, fontweight="bold")

    ax.set_title(f"{model_title}: So sánh Chỉ số Test (v1 vs v2)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Tỉ lệ (%)", fontsize=10)
    ax.set_ylim(89.0, 101.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5, loc="upper right")

plt.tight_layout()
out_file = Path("reports/figures/compare_512_versions.png")
out_file.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_file, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ so sánh: {out_file}")
