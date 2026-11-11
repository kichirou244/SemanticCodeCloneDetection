from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_bcb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tai va trich xuat du lieu BigCloneBench")
    parser.add_argument("--out-dir", type=str, default="data/raw/bigclonebench")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-size", type=int, default=500)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--dev-size", type=int, default=20000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--splits-custom", action="store_true", help="Tai train, validation, test voi so mau tuy chinh")
    parser.add_argument("--train-samples", type=int, default=30000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = BigCloneBenchLoader(cache_dir=out_dir / "hf_cache")

    if args.smoke:
        val = loader.load_from_hf(split="validation", max_samples=args.smoke_size * 4, streaming=True)
        sub = loader.sample_subset(val, n_samples=args.smoke_size, seed=args.seed, balance=True)
        loader.save_parquet(sub, out_dir / "bcb_smoke.parquet")
        loader.save_jsonl(sub, out_dir / "bcb_smoke.jsonl")
        logger.info("Da luu smoke: %d mau", len(sub))

    if args.dev:
        val = loader.load_from_hf(split="validation", max_samples=args.dev_size * 2, streaming=True)
        sub = loader.sample_subset(val, n_samples=args.dev_size, seed=args.seed, balance=True)
        loader.save_parquet(sub, out_dir / "bcb_dev.parquet")
        logger.info("Da luu dev: %d mau", len(sub))

    if args.full:
        for s in ("train", "validation", "test"):
            data = loader.load_from_hf(split=s, streaming=False)
            loader.save_parquet(data, out_dir / f"bcb_{s}.parquet")
            logger.info("Da luu %s: %d mau", s, len(data))

    if args.splits_custom:
        targets = [
            ("train", args.train_samples),
            ("validation", args.val_samples),
            ("test", args.test_samples),
        ]
        for sp, target_count in targets:
            logger.info("Dang tai va can bang split %s (%d mau, muc tieu 50:50)...", sp, target_count)
            half = target_count // 2
            remainder = target_count - half
            positives, negatives = [], []
            for sample in loader.stream_pairs(split=sp):
                if sample.label == 1 and len(positives) < half:
                    positives.append(sample)
                elif sample.label == 0 and len(negatives) < remainder:
                    negatives.append(sample)
                if len(positives) >= half and len(negatives) >= remainder:
                    break
            balanced_sub = positives + negatives
            import random
            rng = random.Random(args.seed)
            rng.shuffle(balanced_sub)
            target_parquet = out_dir / f"bcb_{sp}.parquet"
            loader.save_parquet(balanced_sub, target_parquet)
            pos_cnt = sum(1 for p in balanced_sub if p.label == 1)
            neg_cnt = sum(1 for p in balanced_sub if p.label == 0)
            logger.info("Da luu %s: %s (Tong: %d mau | Pos: %d, Neg: %d)", sp, target_parquet, len(balanced_sub), pos_cnt, neg_cnt)

    if args.split:
        target_count = args.max_samples or 30000
        logger.info("Dang tai va can bang split %s (%d mau)...", args.split, target_count)
        fetch_count = target_count * 2
        raw_samples = loader.load_from_hf(split=args.split, max_samples=fetch_count, streaming=True)
        balanced_sub = loader.sample_subset(raw_samples, n_samples=target_count, seed=args.seed, balance=True)
        target_parquet = out_dir / f"bcb_{args.split}.parquet"
        loader.save_parquet(balanced_sub, target_parquet)
        logger.info("Da luu %s: %s (%d mau, 50:50 balance)", args.split, target_parquet, len(balanced_sub))


if __name__ == "__main__":
    main()
