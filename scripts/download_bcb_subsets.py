"""Script tải và chuẩn bị các tập dữ liệu BigCloneBench (Smoke, Dev, Full).

Cách dùng:
    # 1. Tải tập smoke (500 mẫu) và dev (20.000 mẫu) phục vụ thử nghiệm local
    python scripts/download_bcb_subsets.py --smoke --dev

    # 2. Tải toàn bộ tập dữ liệu (train, validation, test)
    python scripts/download_bcb_subsets.py --full

    # 3. Chỉ tải split cụ thể
    python scripts/download_bcb_subsets.py --split validation --max-samples 1000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Đảm bảo import được src khi chạy script trực tiếp
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_bcb")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải và trích xuất dữ liệu BigCloneBench")
    parser.add_argument("--out-dir", type=str, default="data/raw/bigclonebench", help="Thư mục lưu dữ liệu")
    parser.add_argument("--smoke", action="store_true", help="Tạo tập bcb_smoke (500 mẫu, 50:50)")
    parser.add_argument("--smoke-size", type=int, default=500, help="Số mẫu tập smoke")
    parser.add_argument("--dev", action="store_true", help="Tạo tập bcb_dev (20.000 mẫu)")
    parser.add_argument("--dev-size", type=int, default=20000, help="Số mẫu tập dev")
    parser.add_argument("--full", action="store_true", help="Tải toàn bộ train/validation/test")
    parser.add_argument("--split", type=str, default=None, help="Split cụ thể (train/validation/test)")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = BigCloneBenchLoader(cache_dir=out_dir / "hf_cache")

    logger.info("Khởi động tải dữ liệu BigCloneBench từ Hugging Face: %s", loader.HF_ID)

    if args.smoke:
        logger.info("Đang tạo tập SMOKE (%d mẫu) từ tập validation qua streaming...", args.smoke_size)
        val_samples = loader.load_from_hf(split="validation", max_samples=args.smoke_size * 4, streaming=True)
        smoke_subset = loader.sample_subset(val_samples, n_samples=args.smoke_size, seed=args.seed, balance=True)
        smoke_parquet = out_dir / "bcb_smoke.parquet"
        smoke_jsonl = out_dir / "bcb_smoke.jsonl"
        loader.save_parquet(smoke_subset, smoke_parquet)
        loader.save_jsonl(smoke_subset, smoke_jsonl)
        logger.info("Đã lưu tập smoke: %s (%d mẫu)", smoke_parquet, len(smoke_subset))

    if args.dev:
        logger.info("Đang tạo tập DEV (%d mẫu) từ tập validation qua streaming...", args.dev_size)
        val_samples = loader.load_from_hf(split="validation", max_samples=args.dev_size * 2, streaming=True)
        dev_subset = loader.sample_subset(val_samples, n_samples=args.dev_size, seed=args.seed, balance=True)
        dev_parquet = out_dir / "bcb_dev.parquet"
        loader.save_parquet(dev_subset, dev_parquet)
        logger.info("Đã lưu tập dev: %s (%d mẫu)", dev_parquet, len(dev_subset))

    if args.full:
        for sp in ("train", "validation", "test"):
            logger.info("Đang tải toàn bộ split: %s...", sp)
            samples = loader.load_from_hf(split=sp, max_samples=args.max_samples)
            target_parquet = out_dir / f"bcb_{sp}.parquet"
            loader.save_parquet(samples, target_parquet)
            logger.info("Đã lưu %s: %s (%d mẫu)", sp, target_parquet, len(samples))

    if args.split:
        use_streaming = args.max_samples is not None
        logger.info("Đang tải split: %s (max: %s, streaming=%s)...", args.split, args.max_samples, use_streaming)
        samples = loader.load_from_hf(split=args.split, max_samples=args.max_samples, streaming=use_streaming)
        target_parquet = out_dir / f"bcb_{args.split}.parquet"
        loader.save_parquet(samples, target_parquet)
        logger.info("Đã lưu: %s (%d mẫu)", target_parquet, len(samples))


if __name__ == "__main__":
    main()
