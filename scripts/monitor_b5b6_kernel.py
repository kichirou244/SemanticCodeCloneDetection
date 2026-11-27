#!/usr/bin/env python3
"""Theo dõi kernel B5/B6 trên Kaggle và tải checkpoint/metrics về results/."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-b5-b6-focal-aug-supcon-m1-m2"
DOWNLOAD_DIR = Path("results/b5b6_download")


def status() -> str:
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    return (res.stdout or res.stderr).strip()


def download() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(DOWNLOAD_DIR), "--force"],
        capture_output=True,
        text=True,
    )
    print(res.stdout or res.stderr, flush=True)
    for f in DOWNLOAD_DIR.glob("exp_*_b5b6_*"):
        model = "exp_m1_b5b6" if f.name.startswith("exp_m1_b5b6") else "exp_m2_b5b6"
        dest_dir = Path("results") / model
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f.name[len(model) + 1 :]
        shutil.copy2(f, dest)
        print(f"  [DISPATCH] {dest}", flush=True)
        if dest.name == "test_metrics.json":
            print(f"\n>>> {model} TEST METRICS:\n{dest.read_text(encoding='utf-8')}", flush=True)


def main() -> None:
    print(f"Theo dõi kernel: {KERNEL_SLUG}", flush=True)
    for attempt in range(120):
        s = status()
        print(f"[{attempt:03d}] {s}", flush=True)
        if "COMPLETE" in s:
            print(">>> Kernel hoàn tất, tải kết quả...", flush=True)
            download()
            break
        if any(k in s for k in ("ERROR", "CANCEL", "FAILED")):
            print(">>> Kernel dừng bất thường, thử tải output đã có...", flush=True)
            download()
            break
        time.sleep(60)
    else:
        print(">>> Hết thời gian chờ monitor.", flush=True)


if __name__ == "__main__":
    main()
