from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def get_git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def build_report() -> dict:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": get_git_commit(),
    }
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["cuda_version"] = torch.version.cuda
            report["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        report["torch"] = "not-installed"

    for mod in ("transformers", "datasets", "sklearn", "numpy", "pandas", "yaml"):
        try:
            report[mod] = __import__(mod).__version__
        except Exception:
            report[mod] = "not-installed"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/env.json")
    args = ap.parse_args()
    report = build_report()
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
