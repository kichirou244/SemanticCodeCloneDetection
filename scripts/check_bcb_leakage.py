"""Kiểm tra rò rỉ dữ liệu (Leakage Audit) giữa các split của BigCloneBench Java.

Cách dùng:
    python scripts/check_bcb_leakage.py --train <train_file> --val <val_file> --test <test_file>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.data.validators.pair_audit import check_pair_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_bcb_leakage")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm tra rò rỉ dữ liệu giữa các tập BigCloneBench")
    parser.add_argument("--train", type=str, default=None, help="Đường dẫn file train parquet/jsonl")
    parser.add_argument("--val", type=str, default=None, help="Đường dẫn file val parquet/jsonl")
    parser.add_argument("--test", type=str, default=None, help="Đường dẫn file test parquet/jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Giới hạn số mẫu mỗi split để kiểm tra nhanh")
    parser.add_argument("--output", type=str, default=None, help="Đường dẫn lưu kết quả JSON")
    args = parser.parse_args()

    loader = BigCloneBenchLoader()
    splits = {}

    for name, p_str in (("train", args.train), ("val", args.val), ("test", args.test)):
        if p_str and Path(p_str).exists():
            path = Path(p_str)
            logger.info("Đang nạp split '%s' từ %s...", name, path)
            if path.suffix == ".parquet":
                splits[name] = loader.load_parquet(path, max_samples=args.max_samples)
            else:
                splits[name] = loader.load_jsonl(path, max_samples=args.max_samples)

    if len(splits) < 2:
        logger.warning("Cần cung cấp ít nhất 2 split để kiểm tra chồng lấn leakage (ví dụ --train và --val).")
        return

    logger.info("Đang tiến hành kiểm tra rò rỉ cặp, hàm và hash code...")
    result = check_pair_leakage(splits)

    print("\n" + "=" * 50)
    print("KẾT QUẢ LEAKAGE AUDIT BIGCLONEBENCH:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Đã lưu kết quả leakage vào: %s", out_path)

    if result["has_critical_pair_leakage"]:
        logger.error("CẢNH BÁO: Phát hiện rò rỉ cặp trực tiếp giữa các split!")
        sys.exit(1)
    else:
        logger.info("XÁC NHẬN: Không phát hiện rò rỉ cặp trực tiếp giữa các split (PASS).")


if __name__ == "__main__":
    main()
