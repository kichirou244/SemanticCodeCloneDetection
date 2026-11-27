#!/usr/bin/env python3
"""Theo dõi kernel M1/M2 Best+ và tải kết quả về results/exp_m1_best_plus, exp_m2_best_plus."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SLUG = "onngctrnchu/bcb-m1-m2-best-plus-focal-aug-supcon"
DOWNLOAD_DIR = Path("results/best_plus_download")


def status() -> str:
    r = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def download() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(DOWNLOAD_DIR), "--force"],
        capture_output=True, text=True,
    )
    print(r.stdout or r.stderr, flush=True)
    for exp in ("exp_m1_best_plus", "exp_m2_best_plus"):
        dest_dir = Path("results") / exp
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in DOWNLOAD_DIR.glob(f"{exp}_*"):
            dest = dest_dir / f.name[len(exp) + 1 :]
            shutil.copy2(f, dest)
            print(f"  [DISPATCH] {dest}", flush=True)
            if dest.name == "test_metrics.json":
                print(f"\n>>> {exp} TEST:\n{dest.read_text(encoding='utf-8')}", flush=True)


def main() -> None:
    for i in range(180):
        s = status()
        if i % 5 == 0:
            print(f"[{i:03d}] {s}", flush=True)
        if "COMPLETE" in s:
            download()
            break
        if any(k in s for k in ("ERROR", "CANCEL", "FAILED")):
            print("Kernel dung bat thuong, thu tai output...", flush=True)
            download()
            break
        time.sleep(60)
    else:
        print("Het thoi gian cho monitor.", flush=True)


if __name__ == "__main__":
    main()
