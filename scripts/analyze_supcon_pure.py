#!/usr/bin/env python3
"""So sánh phân bố lớp: DiffMul (không SupCon) vs DiffMul+SupCon thuần, cho M1 và M2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.loaders.bigclonebench import BigCloneBenchLoader  # noqa: E402
from src.postprocess.embeddings import extract_pair_embeddings  # noqa: E402


def cos_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/supcon_pure")
    p.add_argument("--pairs-file", type=Path, default=ROOT / "data/raw/bigclonebench/bcb_test.parquet")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    pairs = BigCloneBenchLoader().load_parquet(args.pairs_file)
    y = np.asarray([int(x.label) for x in pairs], dtype=int)
    print("Pairs:", len(pairs), flush=True)

    models = {
        "M1 DiffMul (no SupCon)": ROOT / "results/ckpts/m1_diffmul_best.pt",
        "M1 DiffMul + SupCon": ROOT / "results/ckpts/m1_diffmul_supcon_best.pt",
        "M2 DiffMul (no SupCon)": ROOT / "results/ckpts/m2_diffmul_best.pt",
        "M2 DiffMul + SupCon": ROOT / "results/ckpts/m2_diffmul_supcon_best.pt",
    }
    report = {"n_pairs": len(pairs), "models": {}}
    data = {}
    for name, path in models.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ha, hb = extract_pair_embeddings(ckpt, pairs, batch_size=args.batch_size,
                                         max_length=512, device=device, use_amp=True, label=name)
        c = cos_sim(ha, hb)
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, c))
        report["models"][name] = {
            "cos_clone_mean": round(float(c[y == 1].mean()), 4),
            "cos_nonclone_mean": round(float(c[y == 0].mean()), 4),
            "auc_cosine_only": round(auc, 4),
        }
        data[name] = (ha, c)
        print(name, report["models"][name], flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        fp = r"C:\Windows\Fonts\times.ttf"
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
        plt.rcParams["axes.unicode_minus"] = False

        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        fig, axes = plt.subplots(4, 2, figsize=(13, 17))
        for row, (name, (ha, c)) in enumerate(data.items()):
            # cột: histogram cosine
            ax = axes[row, 0]
            ax.hist(c[y == 0], bins=40, alpha=0.6, label="Non-clone", color="#d62728", density=True)
            ax.hist(c[y == 1], bins=40, alpha=0.6, label="Clone", color="#1f77b4", density=True)
            ax.set_title(f"{name} — AUC(cos)={report['models'][name]['auc_cosine_only']}", fontsize=10)
            ax.set_xlabel("Cosine"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

            # chấm: t-SNE trên hA (sample tối đa 2000)
            ax = axes[row, 1]
            idx = np.random.default_rng(42).choice(len(y), size=min(2000, len(y)), replace=False)
            Xp = PCA(n_components=min(50, ha.shape[1], len(idx) - 1), random_state=42).fit_transform(ha[idx])
            Z = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Xp)
            ax.scatter(Z[y[idx] == 0, 0], Z[y[idx] == 0, 1], s=6, alpha=0.5, label="Non-clone", color="#d62728")
            ax.scatter(Z[y[idx] == 1, 0], Z[y[idx] == 1, 1], s=6, alpha=0.5, label="Clone", color="#1f77b4")
            ax.set_title(f"{name} — t-SNE của hA", fontsize=10)
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8)
        fig.suptitle("Phân bố CLS: cosine (trái) và t-SNE (phải) — DiffMul không/có SupCon thuần", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "supcon_pure_comparison.png", dpi=150)
        plt.close(fig)
        print("figure saved", flush=True)
    except Exception as e:
        print("figure error:", e, flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
