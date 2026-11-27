#!/usr/bin/env python3
"""Theo dõi kernel selection-opt và tải kết quả về results/exp_selection_opt/."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-selection-opt-val10k-stacking-v3"
DOWNLOAD_DIR = Path("results/selection_opt_download")
OUT_DIR = Path("results/exp_selection_opt")


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
    for f in DOWNLOAD_DIR.glob("sel_*"):
        name = f.name.replace("sel_", "", 1)
        if name == "live_progress.log":
            name = "live_progress.log"
        shutil.copy2(f, OUT_DIR / name)
        print(f"  [DISPATCH] {name}", flush=True)
    report = OUT_DIR / "report.md"
    if report.exists():
        print("\n" + report.read_text(encoding="utf-8"), flush=True)


def main() -> None:
    for i in range(12):
        s = status()
        print(f"[{i:02d}] {s}", flush=True)
        if "COMPLETE" in s:
            download()
            break
        if any(k in s for k in ("ERROR", "CANCEL", "FAILED")):
            print("Kernel dung bat thuong, thu tai output...", flush=True)
            download()
            break
        time.sleep(25)
    else:
        print("Het thoi gian cho, kernel van chay.", flush=True)


if __name__ == "__main__":
    main()
