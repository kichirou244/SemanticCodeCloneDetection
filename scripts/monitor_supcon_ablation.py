#!/usr/bin/env python3
"""Theo dõi 2 kernel ablation SupCon và tải kết quả về results/exp_*_diffmul_supcon/."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

KAGGLE = r"C:\Users\Kichirou\AppData\Local\Programs\Python\Python310\Scripts\kaggle.exe"
JOBS = [
    ("onngctrnchu/bcb-ablation-m1-diffmul-supcon", "exp_m1_diffmul_supcon"),
    ("onngctrnchu/bcb-ablation-m2-diffmul-supcon", "exp_m2_diffmul_supcon"),
]
DL = Path("results/ablation_download")


def st(slug: str) -> str:
    r = subprocess.run([KAGGLE, "kernels", "status", slug], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip()


def dl(slug: str, exp: str) -> None:
    d = DL / exp
    d.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([KAGGLE, "kernels", "output", slug, "-p", str(d), "--force"], capture_output=True, text=True)
    print((r.stdout or r.stderr)[-1500:], flush=True)
    dest = Path("results") / exp
    dest.mkdir(parents=True, exist_ok=True)
    for f in d.glob(f"{exp}_*"):
        name = f.name[len(exp) + 1:]
        if name == "live_progress.log":
            continue
        shutil.copy2(f, dest / name)
        print(f"  [DISPATCH] {dest / name}", flush=True)
    m = dest / "test_metrics.json"
    if m.exists():
        print(f"\n>>> {exp}:\n{m.read_text(encoding='utf-8')}", flush=True)


def main() -> None:
    pending = list(JOBS)
    for i in range(180):
        if i % 5 == 0:
            print(f"[{i:03d}] pending={[e for _, e in pending]}", flush=True)
        done = []
        for slug, exp in pending:
            s = st(slug)
            if "COMPLETE" in s:
                print(f">>> {exp} COMPLETE, tải...", flush=True)
                dl(slug, exp)
                done.append((slug, exp))
            elif any(k in s for k in ("ERROR", "CANCEL", "FAILED")):
                print(f">>> {exp} ERROR, thử tải output...", flush=True)
                dl(slug, exp)
                done.append((slug, exp))
        for x in done:
            pending.remove(x)
        if not pending:
            break
        time.sleep(60)
    else:
        print("Het thoi gian cho monitor.", flush=True)


if __name__ == "__main__":
    main()
