#!/usr/bin/env python3
"""Theo dõi kernel supcon-embedding-viz và tải kết quả về results/supcon_embedding/."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SLUG = "onngctrnchu/bcb-supcon-embedding-visualization"
DOWNLOAD_DIR = Path("results/supcon_viz_download")
OUT_DIR = Path("results/supcon_embedding")


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in DOWNLOAD_DIR.glob("viz_*"):
        name = f.name.replace("viz_", "", 1)
        shutil.copy2(f, OUT_DIR / name)
        print(f"  [DISPATCH] {name}", flush=True)
    report = OUT_DIR / "report.json"
    if report.exists():
        print("\n" + report.read_text(encoding="utf-8"), flush=True)


def main() -> None:
    for i in range(60):
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
        time.sleep(30)
    else:
        print("Het thoi gian cho monitor.", flush=True)


if __name__ == "__main__":
    main()
