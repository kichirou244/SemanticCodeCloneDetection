#!/usr/bin/env python3
"""Tái lập đúng tập test 5k mà notebook Kaggle đã dùng.

Notebook ``bcb_tuning_m1_m2_diffmul`` chạy ``download_bcb_subsets.py --splits-custom``
khi dataset Kaggle không có sẵn parquet. Script đó lấy **2500 clone đầu tiên** và
**2500 non-clone đầu tiên** theo thứ tự stream của split ``test`` trên Hugging Face,
sau đó shuffle bằng seed 42. Kết quả là ``bcb_test.parquet`` mà ``bcb_runner`` nạp
lúc evaluate.

Script này tái lập bước trên và ghi ra JSONL để phân tích lỗi.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "bcb_5k_test_pairs_kaggle.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=5000)
    args = parser.parse_args()

    loader = BigCloneBenchLoader(cache_dir=ROOT / "data" / "raw" / "bigclonebench" / "hf_cache")
    half = args.n // 2
    remainder = args.n - half
    positives, negatives = [], []

    print("Đang stream split test từ Hugging Face...", flush=True)
    for i, sample in enumerate(loader.stream_pairs(split="test")):
        if sample.label == 1 and len(positives) < half:
            positives.append(sample)
        elif sample.label == 0 and len(negatives) < remainder:
            negatives.append(sample)
        if len(positives) >= half and len(negatives) >= remainder:
            break
        if (i + 1) % 20000 == 0:
            print(f"  đã quét {i + 1} cặp | pos={len(positives)} neg={len(negatives)}", flush=True)

    print(f"Thu được {len(positives)} clone + {len(negatives)} non-clone", flush=True)
    pairs = positives + negatives
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for idx, p in enumerate(pairs):
            f.write(
                json.dumps(
                    {
                        "id": p.pair_id or f"{p.id1}:::{p.id2}",
                        "id1": p.id1,
                        "id2": p.id2,
                        "func1": p.func1,
                        "func2": p.func2,
                        "label": int(p.label),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"Đã ghi {len(pairs)} cặp vào {args.out}", flush=True)


if __name__ == "__main__":
    main()
