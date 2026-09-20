"""Báo cáo môi trường tái lập: python scripts/env_report.py (ghi ra results/env.json)."""

import json
import platform
import sys


def main() -> None:
    report = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["cuda_version"] = torch.version.cuda
    except ImportError:
        report["torch"] = "not-installed"
    try:
        import transformers

        report["transformers"] = transformers.__version__
    except ImportError:
        report["transformers"] = "not-installed"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
