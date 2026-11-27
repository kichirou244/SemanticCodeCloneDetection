#!/usr/bin/env python3
"""Trung bình các SWA snapshot (do bcb_runner lưu khi bật `swa.enabled`) thành 1 checkpoint.

Ví dụ config bổ sung:
    swa:
      enabled: true
      start_epoch: 3
      dir: swa

Sau khi train:
    python scripts/average_swa_snapshots.py --ckpt-dir results/exp_m1_diffmul
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess.averaging import average_checkpoint_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Average SWA epoch snapshots")
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=str, default="swa")
    parser.add_argument("--out", type=str, default="swa_checkpoint.pt")
    parser.add_argument("--start-epoch", type=int, default=0)
    args = parser.parse_args()

    snapshot_dir = args.ckpt_dir / args.snapshot_dir
    paths = sorted(p for p in snapshot_dir.glob("epoch_*.pt") if int(p.stem.split("_")[1]) >= args.start_epoch)
    if len(paths) < 2:
        raise RuntimeError(f"Cần ít nhất 2 snapshot trong {snapshot_dir}, tìm thấy {len(paths)}")

    out_path = args.ckpt_dir / args.out
    average_checkpoint_paths(paths, out_path)
    print(f"Đã lưu {out_path} từ {len(paths)} snapshot: {[p.name for p in paths]}")


if __name__ == "__main__":
    main()
