"""Audit dataset BigCloneBench Java: python scripts/audit_bcb_dataset.py."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.data.validators.pair_audit import audit_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_bcb")


def generate_markdown_report(report: dict, title: str = "BigCloneBench Audit Report") -> str:
    """Định dạng kết quả audit thành tài liệu Markdown chuẩn chỉnh."""
    stats = report.get("length_stats", {})
    return f"""# {title}

## 1. Tổng quan phân phối nhãn
- **Tổng số cặp đánh giá:** {report['n_pairs']:,}
- **Cặp Clone (Nhãn 1):** {report['n_positive']:,} ({report['positive_ratio'] * 100:.2f}%)
- **Cặp Non-clone (Nhãn 0):** {report['n_negative']:,} ({(1 - report['positive_ratio']) * 100:.2f}%)
- **Số hàm (functions) duy nhất:** {report['unique_func_ids']:,}
- **Đoạn code rỗng (missing code):** {report['missing_code']}
- **Cặp trùng lặp (duplicate pairs):** {report['duplicate_pairs']}
- **Cặp xung đột nhãn (conflicting labels):** {report['conflicting_pairs']}

## 2. Thống kê độ dài mã nguồn (Code Length Statistics)
| Chỉ số | Dòng code (Lines) | Ký tự (Characters) |
| :--- | :---: | :---: |
| **Nhỏ nhất (Min)** | {stats.get('lines_min', 0)} | {stats.get('chars_min', 0)} |
| **Trung vị (Median)** | {stats.get('lines_median', 0)} | {stats.get('chars_median', 0)} |
| **Trung bình (Mean)** | {stats.get('lines_mean', 0):.2f} | {stats.get('chars_mean', 0):.2f} |
| **Lớn nhất (Max)** | {stats.get('lines_max', 0)} | {stats.get('chars_max', 0)} |

## 3. Đánh giá chất lượng dữ liệu
> **Trạng thái:** {"[PASS] Dữ liệu hợp lệ, không có xung đột nhãn" if report['conflicting_pairs'] == 0 and report['missing_code'] == 0 else "[WARN] Cần kiểm tra lại các trường thiếu hoặc xung đột"}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dữ liệu BigCloneBench Java")
    parser.add_argument("--file", type=str, default=None, help="Đường dẫn file parquet/jsonl local")
    parser.add_argument("--split", type=str, default="validation", help="Split từ HF nếu không nạp file")
    parser.add_argument("--max-samples", type=int, default=1000, help="Số mẫu tối đa để nạp từ HF")
    parser.add_argument("--output-json", type=str, default=None, help="Lưu report dưới dạng JSON")
    parser.add_argument("--output-md", type=str, default="reports/bcb_data_audit.md", help="Lưu report Markdown")
    args = parser.parse_args()

    loader = BigCloneBenchLoader()

    if args.file and Path(args.file).exists():
        path = Path(args.file)
        logger.info("Đang nạp dữ liệu từ file local: %s...", path)
        if path.suffix == ".parquet":
            pairs = loader.load_parquet(path, max_samples=args.max_samples)
        else:
            pairs = loader.load_jsonl(path, max_samples=args.max_samples)
    else:
        logger.info("Đang nạp %d mẫu từ split '%s' qua Hugging Face...", args.max_samples, args.split)
        pairs = loader.load_from_hf(split=args.split, max_samples=args.max_samples)

    logger.info("Tiến hành phân tích audit trên %d cặp...", len(pairs))
    report = audit_pairs(pairs)

    print("\n" + "=" * 50)
    print("KẾT QUẢ AUDIT BIGCLONEBENCH:")
    print("=" * 50)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Đã lưu report JSON vào: %s", out_json_path)

    if args.output_md:
        out_md_path = Path(args.output_md)
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        md_content = generate_markdown_report(report)
        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Đã lưu report Markdown vào: %s", out_md_path)


if __name__ == "__main__":
    main()
