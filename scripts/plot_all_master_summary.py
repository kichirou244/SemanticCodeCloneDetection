#!/usr/bin/env python3
"""Tạo biểu đồ tổng hợp Master Summary bao gồm toàn bộ các mô hình và 14 phiên bản tinh chỉnh siêu tham số."""

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

model_keys = [
    ("M1 512 v2", "M1_512_v2"),
    ("M1-LR (1e-5)", "M1_Tuning_LR"),
    ("M1-BS (64)", "M1_Tuning_BS"),
    ("M1-WD (0.1)", "M1_Tuning_WD"),
    ("M2 512 v2", "M2_512_v2"),
    ("M2-DR (0.3)", "M2_Tuning_DR"),
    ("M2-LR (1e-5)", "M2_Tuning_LR"),
    ("M2-BS (64)", "M2_Tuning_BS"),
    ("M3 Opt (8 Ep)", "M3_Opt"),
    ("M3-EP (20 Ep)", "M3_Tuning_EP"),
    ("M3-LR (1e-4)", "M3_Tuning_LR"),
    ("M3-BS (64)", "M3_Tuning_BS"),
    ("M3-HD (300)", "M3_Tuning_HD"),
    ("M4 Opt (8 Ep)", "M4_Opt"),
    ("M4-EP (20 Ep)", "M4_Tuning_EP"),
    ("M4-BS (64)", "M4_Tuning_BS"),
    ("M4-HD (8h)", "M4_Tuning_HD"),
    ("M4-LR (1e-4)", "M4_Tuning_LR"),
]

labels = [k[0] for k in model_keys]
f1_scores = []
acc_scores = []

for _, k in model_keys:
    opt = data[k].get("optimal", data[k].get("fixed", data[k]))
    f1 = float(opt.get("f1", 0.0))
    f1 = f1 if f1 <= 1.0 else f1 / 100.0
    acc = float(opt.get("accuracy", opt.get("acc", 0.0)))
    acc = acc if acc <= 1.0 else acc / 100.0
    f1_scores.append(f1 * 100)
    acc_scores.append(acc * 100)

x = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(18, 6.5), dpi=300)
rects1 = ax.bar(x - width/2, f1_scores, width, label="F1-Score (%)", color="#1f77b4", alpha=0.9, edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, acc_scores, width, label="Accuracy (%)", color="#2ca02c", alpha=0.9, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Tỉ lệ (%)", fontsize=11, fontweight="bold")
ax.set_title("Đối chiếu F1-Score và Accuracy giữa các Phiên bản Baseline và Tinh chỉnh Siêu tham số (BigCloneBench 5.000 cặp)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
ax.set_ylim(50, 105)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
ax.legend(fontsize=10.5, loc="upper right")

for bar in rects1:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

for bar in rects2:
    h = bar.get_height()
    ax.annotate(f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
out_master = FIG_DIR / "all_versions_master_summary.png"
plt.savefig(out_master, dpi=300)
plt.close()
print(f"Da cap nhat thanh cong: {out_master}")
