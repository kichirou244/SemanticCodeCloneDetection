"""Leakage audit: python scripts/check_leakage.py --dataset poj104."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.seed import set_global_seed
from src.data.leakage.checks import run_leakage_audit
from src.data.loaders.codenet_python import CodeNetPythonLoader
from src.data.loaders.poj104 import POJ104Loader
from src.data.splitting.problem_split import split_by_problem

LOADERS = {"poj104": POJ104Loader, "codenet_python": CodeNetPythonLoader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(LOADERS), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--shuffle-buffer", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(args.seed)

    samples = LOADERS[args.dataset]().load(
        split=args.split,
        streaming=args.streaming,
        max_samples=args.max_samples,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )
    # 1) Kiểm tra split CHÍNH THỨC của dataset (theo s.split gốc từ loader):
    #    cùng problem / cùng code có xuyên train/val/test không?
    from collections import defaultdict

    official: dict[str, list] = defaultdict(list)
    for s in samples:
        official[s.split or "unknown"].append(s)
    print("== official splits ==")
    print(json.dumps(run_leakage_audit(dict(official)), indent=2))
    # 2) Kiểm tra problem_split của project (phải pass).
    splits = split_by_problem(samples, seed=args.seed)
    print("== problem_split ==")
    print(json.dumps(run_leakage_audit(splits), indent=2))


if __name__ == "__main__":
    main()
