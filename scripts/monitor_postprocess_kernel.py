#!/usr/bin/env python3
"""Theo dõi kernel postprocess trên Kaggle và tải kết quả về khi hoàn tất."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KERNEL_SLUG = "tranthikhanhlinh1709/bcb-diffmul-postprocess-a1-a2-a3"
DOWNLOAD_DIR = Path("results/postprocess_download")
OUT_DIR = Path("results/exp_diffmul_postprocess")


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in DOWNLOAD_DIR.glob("postprocess_*"):
        clean_name = f.name.replace("postprocess_", "", 1)
        if clean_name == "live_progress.log":
            continue
        shutil.copy2(f, OUT_DIR / clean_name)
        print(f"  [DISPATCH] {clean_name}", flush=True)

    report = OUT_DIR / "postprocess_report.md"
    if report.exists():
        print("\n" + "=" * 70, flush=True)
        print(report.read_text(encoding="utf-8"), flush=True)


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
            print(">>> Kernel dừng bất thường, vẫn thử tải output đã có...", flush=True)
            download()
            break
        time.sleep(60)
    else:
        print(">>> Hết thời gian chờ monitor.", flush=True)


if __name__ == "__main__":
    main()
