"""Audit status CodeNet: python scripts/audit_status.py --max-samples 50000.

Đo phân bố accepted/rejected/other TRƯỚC khi quyết DEC-012 (Step 9).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.common.seed import set_global_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=50000)
    ap.add_argument("--split", default="small")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seed(args.seed)

    ds = load_dataset("windchimeran/codenet_python", split=args.split, streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)
    statuses: Counter = Counter()
    langs: Counter = Counter()
    n = 0
    for row in ds:
        if n >= args.max_samples:
            break
        statuses[str(row.get("status", "?"))] += 1
        langs[str(row.get("language", "?"))] += 1
        n += 1
    print(json.dumps(
        {"n": n, "status": dict(statuses), "language": dict(langs)}, indent=2
    ))


if __name__ == "__main__":
    main()
