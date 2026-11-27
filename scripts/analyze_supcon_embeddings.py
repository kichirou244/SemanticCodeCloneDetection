#!/usr/bin/env python3
"""So sánh phân bố lớp giữa checkpoint KHÔNG có SupCon và CÓ SupCon.

Chỉ phân tích/đánh giá (không tune). Chạy trên Kaggle GPU hoặc local có GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.loaders.bigclonebench import BigCloneBenchLoader  # noqa: E402
from src.postprocess.embeddings import extract_pair_embeddings  # noqa: E402


def cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a * b, axis=1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-supcon", type=Path, required=True, help="Checkpoint không dùng SupCon")
    parser.add_argument("--with-supcon", type=Path, required=True, help="Checkpoint có dùng SupCon")
    parser.add_argument("--pairs-file", type=Path, default=ROOT / "data/raw/bigclonebench/bcb_test.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/supcon_embedding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    pairs = BigCloneBenchLoader().load_parquet(args.pairs_file, max_samples=args.limit)
    y = np.asarray([int(p.label) for p in pairs], dtype=int)
    print(f"Pairs: {len(pairs)} (clone={int(y.sum())}, non-clone={int((1-y).sum())})", flush=True)

    models = {
        "no_supcon": args.no_supcon,
        "with_supcon": args.with_supcon,
    }
    embed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ha, hb = extract_pair_embeddings(
            ckpt, pairs, batch_size=args.batch_size, max_length=args.max_length,
            device=device, use_amp=True, label=name,
        )
        np.save(out_dir / f"{name}_ha.npy", ha)
        np.save(out_dir / f"{name}_hb.npy", hb)
        embed[name] = (ha, hb)

    report: dict[str, Any] = {"n_pairs": len(pairs), "models": {}}
    for name, (ha, hb) in embed.items():
        c = cos_sim(ha, hb)
        auc = float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, c)
        )
        report["models"][name] = {
            "cos_clone_mean": round(float(c[y == 1].mean()), 4),
            "cos_nonclone_mean": round(float(c[y == 0].mean()), 4),
            "cos_gap": round(float(c[y == 1].mean() - c[y == 0].mean()), 4),
            "auc_cosine_only": round(auc, 4),
        }
        print(name, report["models"][name], flush=True)

    # Figure (lon theo sklearn, nếu thiếu thì vẫn lưu report)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for row, (name, (ha, hb)) in enumerate(embed.items()):
            c = cos_sim(ha, hb)
            # (row,0): histogram cosine theo lớp
            ax = axes[row, 0]
            ax.hist(c[y == 0], bins=40, alpha=0.6, label="Non-clone", color="#d62728", density=True)
            ax.hist(c[y == 1], bins=40, alpha=0.6, label="Clone", color="#1f77b4", density=True)
            ax.set_title(f"{name}: cos(hA,hB) — AUC={report['models'][name]['auc_cosine_only']}")
            ax.set_xlabel("Cosine"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

            # (row,1): t-SNE trên hA (sample tối đa 2000)
            ax = axes[row, 1]
            idx = np.random.default_rng(42).choice(len(y), size=min(2000, len(y)), replace=False)
            X = ha[idx]
            n_comp = min(50, X.shape[1], X.shape[0] - 1)
            Xp = PCA(n_components=n_comp, random_state=42).fit_transform(X)
            Z = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(Xp)
            ax.scatter(Z[y[idx] == 0, 0], Z[y[idx] == 0, 1], s=6, alpha=0.5, label="Non-clone", color="#d62728")
            ax.scatter(Z[y[idx] == 1, 0], Z[y[idx] == 1, 1], s=6, alpha=0.5, label="Clone", color="#1f77b4")
            sil = silhouette_score(Z, y[idx]) if len(np.unique(y[idx])) > 1 else float("nan")
            ax.set_title(f"{name}: t-SNE của hA — silhouette={sil:.3f}")
            ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(fontsize=8)
            report["models"][name]["tsne_silhouette"] = round(float(sil), 4)

        fig.suptitle("Phân bố lớp: không có SupCon vs có SupCon", fontsize=13)
        fig.tight_layout()
        fig_path = out_dir / "supcon_embedding_comparison.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Figure: {fig_path}", flush=True)
    except Exception as e:
        print(f"[warn] Không vẽ được figure: {e}", flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
