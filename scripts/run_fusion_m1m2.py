#!/usr/bin/env python3
"""Chạy feature-level fusion M1+M2: trích embedding, train head MLP, đánh giá."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader  # noqa: E402
from src.postprocess.embeddings import extract_pair_embeddings  # noqa: E402
from src.postprocess.fusion import build_fusion_features, train_fusion_head, FusionHead  # noqa: E402
from src.postprocess.calibration import best_f1_threshold  # noqa: E402
from src.postprocess.metrics import compute_metrics, bootstrap_f1, mcnemar  # noqa: E402

CKPTS = {
    "m1": ROOT / "results/exp_m1_lr/best_checkpoint.pt",
    "m2": ROOT / "results/exp_m2_dr/best_checkpoint.pt",
}
BASELINES = {
    "m1": (ROOT / "results/exp_m1_lr/test_metrics.json", 0.9668),
    "m2": (ROOT / "results/exp_m2_dr/test_metrics.json", 0.9707),
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/exp_fusion_m1m2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    loader = BigCloneBenchLoader()
    splits = {
        "train": loader.load_parquet(ROOT / "data/raw/bigclonebench/bcb_train.parquet"),
        "val": loader.load_parquet(ROOT / "data/raw/bigclonebench/bcb_validation.parquet"),
        "test": loader.load_parquet(ROOT / "data/raw/bigclonebench/bcb_test.parquet"),
    }
    y = {s: np.asarray([p.label for p in pairs], dtype=int) for s, pairs in splits.items()}
    print({s: len(p) for s, p in splits.items()}, flush=True)

    emb: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for key, ckpt in CKPTS.items():
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        emb[key] = {}
        for split, pairs in splits.items():
            cache_a = out_dir / f"{key}_{split}_ha.npy"
            cache_b = out_dir / f"{key}_{split}_hb.npy"
            if cache_a.exists() and cache_b.exists():
                ha, hb = np.load(cache_a), np.load(cache_b)
            else:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                ha, hb = extract_pair_embeddings(
                    state, pairs, batch_size=args.batch_size, max_length=args.max_length,
                    device=device, use_amp=True, label=f"{key}/{split}",
                )
                np.save(cache_a, ha); np.save(cache_b, hb)
            emb[key][split] = (ha, hb)

    X = {s: build_fusion_features(emb["m1"][s][0], emb["m1"][s][1], emb["m2"][s][0], emb["m2"][s][1]) for s in splits}
    print({s: X[s].shape for s in X}, flush=True)

    model, info = train_fusion_head(
        X["train"], y["train"], X["val"], y["val"], epochs=args.epochs, seed=args.seed, device=device
    )
    torch.save(model.state_dict(), out_dir / "fusion_head.pt")

    def probs(split: str) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.tensor(X[split], dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()

    p_val, p_test = probs("val"), probs("test")
    thr, val_f1 = best_f1_threshold(y["val"], p_val)
    metrics = compute_metrics(y["test"], p_test, thr)
    ci = bootstrap_f1(y["test"], p_test, thr, seed=args.seed)

    # So sánh với baseline M2 Best (dùng prob từ file test nếu có; ở đây dùng lại score từ model gốc qua logits không có -> tính từ checkpoint)
    report = {
        "protocol": "freeze M1/M2 encoder, concat interaction features, train MLP head; threshold on val F1",
        "val_f1": round(float(val_f1), 4),
        "threshold": round(float(thr), 4),
        "test": metrics,
        "test_f1_bootstrap": ci,
        "train_info": info,
    }
    np.save(out_dir / "fusion_test_probs.npy", p_test)

    # McNemar vs baseline M2 Best: dùng logits M2 đã trích ở trên thay vì chạy lại model
    from src.postprocess.inference import infer_logits

    base_ckpt = CKPTS["m2"]
    base_logits_path = out_dir / "m2_best_test_logits.npy"
    if base_logits_path.exists():
        base_logits = np.load(base_logits_path)
    else:
        base_logits = infer_logits(
            base_ckpt, splits["test"], batch_size=args.batch_size, max_length=args.max_length,
            device=device, use_amp=True, label="m2_baseline/test",
        )
        np.save(base_logits_path, base_logits)
    p_base = torch.sigmoid(torch.tensor(base_logits, dtype=torch.float32)).numpy()
    # M2 Best threshold (exp_m2_dr): chốt từ val = 0.52; không đọc file metrics trong kernel.
    t_thr = 0.52
    report["mcnemar_vs_m2_best"] = mcnemar(y["test"], p_base, t_thr, p_test, thr)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "train_info"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
