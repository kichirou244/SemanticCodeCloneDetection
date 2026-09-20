"""Audit dataset: python scripts/audit_dataset.py --dataset poj104."""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.seed import set_global_seed
from src.data.deduplication.duplicates import duplicate_stats
from src.data.loaders.codenet_python import CodeNetPythonLoader
from src.data.loaders.poj104 import POJ104Loader
from src.data.validators.schema import audit_samples

LOADERS = {"poj104": POJ104Loader, "codenet_python": CodeNetPythonLoader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--split", default=None, help="Chỉ audit split này (vd small)")
    ap.add_argument("--streaming", action="store_true", help="Streaming, tránh arrow cache")
    ap.add_argument("--shuffle-buffer", type=int, default=0)
    ap.add_argument("--output", default=None, help="Lưu full report JSON vào file")
    args = ap.parse_args()
    set_global_seed(args.seed)

    samples = LOADERS[args.dataset]().load(
        split=args.split,
        streaming=args.streaming,
        max_samples=args.max_samples,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    if args.max_samples:
        # Dataset sắp xếp theo problem (vd 200 mẫu đầu cùng 1 problem),
        # nên phải shuffle trước khi slice để tránh sampling bias.
        rng = random.Random(args.seed)
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        samples = [samples[i] for i in idx[: args.max_samples]]
    report = audit_samples(samples)
    report["duplicates"] = duplicate_stats([s.code for s in samples])
    report["first_row_keys"] = (samples[0].metadata or {}).get("row_keys", [])
    report["schema"] = LOADERS[args.dataset]().describe_schema()
    if args.output:
        # Lưu full (kèm problem_counts chi tiết) để làm paper trail.
        import os

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"saved: {args.output}")
    # stdout chỉ in bản gọn (problem_counts hàng trăm keys sẽ nuốt output).
    short = {k: v for k, v in report.items() if k != "problem_counts"}
    print(json.dumps(short, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
