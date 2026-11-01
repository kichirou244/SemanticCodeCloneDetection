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


if __name__ == "__main__":
    main()
