#!/usr/bin/env python3
"""Tạo lại biểu đồ tổng hợp 12 mô hình và so sánh v1 vs v2 từ dữ liệu kiểm thử thực tế chuẩn 1:1."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results" / "all_final_5k_test_results.json"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Biểu đồ tổng hợp Master Summary
model_keys = [
    ("M1 Base", "M1_Base"),
    ("M1 384", "M1_384"),
    ("M1 512 v1", "M1_512_v1"),
    ("M1 512 v2", "M1_512_v2"),
    ("M2 Base", "M2_Base"),
    ("M2 384", "M2_384"),
    ("M2 512 v1", "M2_512_v1"),
    ("M2 512 v2", "M2_512_v2"),
    ("M3 Base", "M3_Base"),
    ("M3 Opt", "M3_Opt"),
    ("M4 Base", "M4_Base"),
    ("M4 Opt", "M4_Opt"),
]

labels = [k[0] for k in model_keys]
f1_scores = []
acc_scores = []

for _, k in model_keys:
    opt = data[k].get("optimal", data[k])
    f1_scores.append(float(opt.get("f1", 0.0)))
    acc_scores.append(float(opt.get("acc", 0.0)))

x = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(14.5, 5.5), dpi=300)
rects1 = ax.bar(x - width/2, f1_scores, width, label="F1-Score (%)", color="#1f77b4", alpha=0.9, edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, acc_scores, width, label="Accuracy (%)", color="#2ca02c", alpha=0.9, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Tỉ lệ (%)", fontsize=11, fontweight="bold")
ax.set_title("Hiệu năng Kiểm thử Thực tế trên Tập BigCloneBench Test (5.000 cặp — Tỉ lệ 1:1 Chuẩn)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10, fontweight="bold")
ax.set_ylim(40, 105)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
ax.legend(fontsize=10.5, loc="upper right")

for bar in rects1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

for bar in rects2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

plt.tight_layout()
fig_path1 = FIG_DIR / "all_versions_master_summary.png"
plt.savefig(fig_path1, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ: {fig_path1}")

# 2. Biểu đồ so sánh v1 vs v2 cho 512 tokens
metrics_keys = ["f1", "acc", "prec", "rec", "roc_auc"]
metrics_labels = ["F1-Score", "Accuracy", "Precision", "Recall", "ROC-AUC"]

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)
comparisons = [
    ("M1 (CodeBERT 512)", "M1_512_v1", "M1_512_v2"),
    ("M2 (GraphCodeBERT 512)", "M2_512_v1", "M2_512_v2"),
]
colors = ["#1f77b4", "#d95f02"]
x_m = np.arange(len(metrics_keys))
b_w = 0.35

for ax_idx, (title, k_v1, k_v2) in enumerate(comparisons):
    ax = axes[ax_idx]
    v1_vals = [float(data[k_v1]["optimal"].get(k, 0)) for k in metrics_keys]
    v2_vals = [float(data[k_v2]["optimal"].get(k, 0)) for k in metrics_keys]

    b1 = ax.bar(x_m - b_w/2, v1_vals, width=b_w, label="v1 (Tiêu chuẩn 3ep)", color=colors[0], alpha=0.88, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x_m + b_w/2, v2_vals, width=b_w, label="v2 (Mở rộng 12-13ep)", color=colors[1], alpha=0.88, edgecolor="black", linewidth=0.8)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title(f"{title}: So sánh Chỉ số Test (v1 vs v2)", fontsize=11, fontweight="bold")
    ax.set_xticks(x_m)
    ax.set_xticklabels(metrics_labels, fontsize=10, fontweight="bold")
    ax.set_ylabel("Tỉ lệ (%)", fontsize=10)
    ax.set_ylim(88.0, 102.0)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9.5, loc="upper right")

plt.tight_layout()
fig_path2 = FIG_DIR / "compare_512_versions.png"
plt.savefig(fig_path2, dpi=300)
plt.close()
print(f"Đã lưu biểu đồ: {fig_path2}")
