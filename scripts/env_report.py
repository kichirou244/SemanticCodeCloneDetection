"""Báo cáo môi trường tái lập: python scripts/env_report.py [--output results/env.json].

Ghi kèm git commit + timestamp để run nào cũng truy được về đúng môi trường
(tài liệu reproducibility yêu cầu artifact này).
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def build_report() -> dict:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["cuda_version"] = torch.version.cuda
            report["gpu_count"] = torch.cuda.device_count()
            report["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:
        report["torch"] = "not-installed"
    for mod in ("transformers", "datasets", "sklearn", "numpy", "scipy", "pandas", "yaml"):
        try:
            report[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            report[mod] = "not-installed"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/env.json")
    args = ap.parse_args()
    report = build_report()
    print(json.dumps(report, indent=2))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nđã ghi {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
