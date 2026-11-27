#!/usr/bin/env python3
"""Chạy A1 (calibration + OOF threshold) → A2 (SWA best+last) → A3 (ensemble M1+M2).

Chạy trên Kaggle (GPU) sau khi đã có checkpoint trong `results/exp_m1_diffmul` và
`results/exp_m2_diffmul`. Mọi ngưỡng/trọng số được chốt trên validation; test chỉ
để đánh giá cuối kèm bootstrap CI và McNemar.

Ví dụ:
    python scripts/postprocess_diffmul.py \
        --val-file data/raw/bigclonebench/bcb_validation.parquet \
        --test-file data/raw/bigclonebench/bcb_test.parquet \
        --out-dir results/exp_diffmul_postprocess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.postprocess.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Postprocess M1/M2 DiffMul: calibration + SWA + ensemble")
    parser.add_argument("--val-file", type=Path, default=ROOT / "data/raw/bigclonebench/bcb_validation.parquet")
    parser.add_argument("--test-file", type=Path, default=ROOT / "data/raw/bigclonebench/bcb_test.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/exp_diffmul_postprocess")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--swa-alpha", type=float, default=0.5, help="Trọng số của best checkpoint khi trộn SWA")
    parser.add_argument("--limit-val", type=int, default=None, help="Smoke test: giới hạn số cặp validation")
    parser.add_argument("--limit-test", type=int, default=None, help="Smoke test: giới hạn số cặp test")
    parser.add_argument("--m1-best", type=Path, default=None, help="Override checkpoint M1 best")
    parser.add_argument("--m1-last", type=Path, default=None, help="Override checkpoint M1 last")
    parser.add_argument("--m2-best", type=Path, default=None, help="Override checkpoint M2 best")
    parser.add_argument("--m2-last", type=Path, default=None, help="Override checkpoint M2 last")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    for path in (args.val_file, args.test_file):
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file dữ liệu: {path}")

    report = run_pipeline(args)
    print(f"\n>>> Xong. Báo cáo: {(Path(args.out_dir) / 'postprocess_report.md').resolve()}", flush=True)
    print(f">>> Ensemble test F1 = {report['ensemble']['test']['f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
