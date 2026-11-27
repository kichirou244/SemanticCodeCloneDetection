#!/usr/bin/env python3
"""Tối ưu protocol chọn ngưỡng/trọng số trên validation lớn (10k) + stacking.

Chạy trên Kaggle GPU. Input:
  - validation_shard_0.parquet (dataset riêng)
  - M1 gốc + M2 B5+B6 best checkpoint (2 dataset Kaggle)
Output:
  - results/exp_selection_opt/report.json / report.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader  # noqa: E402
from src.postprocess.inference import infer_logits, load_cache, save_cache  # noqa: E402
from src.postprocess.selection import (  # noqa: E402
    apply_stacking,
    ensemble_oof,
    evaluate_system,
    isotonic_oof,
    compare_significance,
    stacking_oof,
)

MODEL_SPECS = {
    "m1": {"label": "M1 CodeBERT DiffMul goc", "ckpt": "results/exp_m1_diffmul/best_checkpoint.pt"},
    "m2": {"label": "M2 GraphCodeBERT B5+B6", "ckpt": "results/exp_m2_b5b6/best_checkpoint.pt"},
}


def load_shard_balanced(path: Path, n_total: int, seed: int) -> list[Any]:
    """Lấy n_total/2 clone đầu + n_total/2 non-clone đầu từ shard, shuffle."""
    df = pd.read_parquet(path)
    half = n_total // 2
    pos = df[df["label"].astype(bool)].head(half)
    neg = df[~df["label"].astype(bool)].head(half)
    print(f"[data] shard={len(df)} | sampled {len(pos)} pos + {len(neg)} neg", flush=True)
    if len(pos) < half or len(neg) < half:
        raise RuntimeError("Khong du mau can bang trong validation shard")

    loader = BigCloneBenchLoader()
    pairs = []
    for _, row in pd.concat([pos, neg]).iterrows():
        pairs.append(
            loader.parse_row(
                {
                    "id": row.get("id"),
                    "id1": row["id1"],
                    "id2": row["id2"],
                    "func1": row["func1"],
                    "func2": row["func2"],
                    "label": bool(row["label"]),
                },
                split="validation",
            )
        )
    random.Random(seed).shuffle(pairs)
    return pairs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, default=ROOT / "data/raw/bigclonebench/validation_shard_0.parquet")
    parser.add_argument("--test-file", type=Path, default=ROOT / "data/raw/bigclonebench/bcb_test.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/exp_selection_opt")
    parser.add_argument("--n-val", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    out_dir = args.out_dir
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    val_pairs = load_shard_balanced(args.shard, args.n_val, args.seed)
    test_pairs = BigCloneBenchLoader().load_parquet(args.test_file)
    y_val = np.array([p.label for p in val_pairs], dtype=int)
    y_test = np.array([p.label for p in test_pairs], dtype=int)
    print(f"val={len(val_pairs)} test={len(test_pairs)}", flush=True)

    logits: dict[str, dict[str, np.ndarray]] = {}
    for key, spec in MODEL_SPECS.items():
        ckpt_path = ROOT / spec["ckpt"]
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        logits[key] = {}
        for split, pairs in (("val", val_pairs), ("test", test_pairs)):
            cache = cache_dir / f"{key}_{split}_logits.npy"
            arr = load_cache(cache)
            if arr is None:
                arr = infer_logits(
                    ckpt_path, pairs, batch_size=args.batch_size, max_length=args.max_length,
                    device=device, use_amp=True, label=f"{key}/{split}",
                )
                save_cache(cache, arr)
            logits[key][split] = arr

    za, zb = logits["m1"]["val"], logits["m2"]["val"]
    ta, tb = logits["m1"]["test"], logits["m2"]["test"]

    # 1) Calibration isotonic OOF cho từng model
    oa = isotonic_oof(za, y_val, args.n_folds, args.seed)
    ob = isotonic_oof(zb, y_val, args.n_folds, args.seed)

    # 2) Hệ thống: M2 đơn, M1 đơn, ensemble-average weight grid, stacking
    systems: dict[str, dict[str, Any]] = {}
    # M2 single: dùng OOF ob và test prob từ isotonic fit trên toàn val
    from src.postprocess.calibration import fit_isotonic, apply_calibration
    p2_test = apply_calibration(tb, "isotonic", fit_isotonic(zb, y_val))
    systems["m2_single"] = evaluate_system("M2 B5B6 + isotonic", y_val, ob, y_test, p2_test, args.n_boot, args.seed)

    p1_test = apply_calibration(ta, "isotonic", fit_isotonic(za, y_val))
    systems["m1_single"] = evaluate_system("M1 goc + isotonic", y_val, oa, y_test, p1_test, args.n_boot, args.seed)

    best_w, mix_val, mix_f1 = ensemble_oof(oa, ob, y_val, args.n_folds, args.seed)
    mix_test = best_w * p1_test + (1.0 - best_w) * p2_test
    systems["ensemble_avg"] = evaluate_system(
        f"Ensemble avg (w={best_w:.2f})", y_val, mix_val, y_test, mix_test, args.n_boot, args.seed
    )

    stack_val = stacking_oof(za, zb, y_val, args.n_folds, args.seed)
    stack_test = apply_stacking(ta, tb, za, zb, y_val)
    systems["stacking"] = evaluate_system("Stacking logistic", y_val, stack_val, y_test, stack_test, args.n_boot, args.seed)

    report: dict[str, Any] = {
        "protocol": {
            "val_source": "validation_shard_0.parquet (first 5k pos + 5k neg, shuffle seed 42)",
            "n_val": len(val_pairs), "n_test": len(test_pairs),
            "oof_folds": args.n_folds, "seed": args.seed,
            "test_use": "chi danh gia cuoi; khong tune ngưỡng/trong so tren test",
        },
        "systems": {},
    }
    for name, sysinfo in systems.items():
        clean = {k: v for k, v in sysinfo.items() if not k.endswith("_prob")}
        report["systems"][name] = clean

    report["significance"] = {
        "stacking_vs_m2": compare_significance(y_test, systems["m2_single"], systems["stacking"]),
        "ensemble_vs_m2": compare_significance(y_test, systems["m2_single"], systems["ensemble_avg"]),
    }

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(_render(report), encoding="utf-8")
    print("\n" + _render(report), flush=True)
    np.save(out_dir / "stacking_test_probs.npy", stack_test)
    np.save(out_dir / "ensemble_test_probs.npy", mix_test)


def _render(report: dict[str, Any]) -> str:
    lines = ["# Tối ưu protocol chọn ngưỡng/trọng số", ""]
    p = report["protocol"]
    lines.append(f"- Validation: {p['n_val']:,} cặp từ `{p['val_source']}` | Test: {p['n_test']:,} cặp")
    lines.append(f"- OOF: {p['oof_folds']} fold | {p['test_use']}")
    lines.append("")
    lines.append("| Hệ thống | Ngưỡng | Val OOF F1 | Test F1 | P | R | FP/FN | CI95 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for name, s in report["systems"].items():
        m, ci = s["test"], s["test_f1_bootstrap"]
        lines.append(
            f"| {name} | {s['threshold']:.2f} | {s['oof_f1']:.4f} | **{m['f1']:.4f}** | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['fp']}/{m['fn']} | "
            f"[{ci['ci95'][0]:.4f}, {ci['ci95'][1]:.4f}] |"
        )
    lines.append("")
    for k, v in report["significance"].items():
        lines.append(f"- McNemar `{k}`: p={v['p_value']:.4g}, discordant={v['discordant']}, significant={v['significant']}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
