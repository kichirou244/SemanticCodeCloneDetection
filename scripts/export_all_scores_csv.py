#!/usr/bin/env python3
"""Xuất ``m1_all_scores.csv`` / ``m2_all_scores.csv`` từ all_scores.jsonl.

Thay vì độ dài ký tự (len1/len2), mỗi cặp được gắn số token thực tế của từng hàm
(đếm bằng tokenizer của chính mô hình, ``add_special_tokens=False``, không cắt bỏ).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = [
    "model_label",
    "error_type",
    "index",
    "pair_id",
    "id1",
    "id2",
    "label",
    "prob",
    "cosine",
    "threshold",
    "token1",
    "token2",
]

GROUP_ORDER = {"TP": 0, "TN": 1, "FP": 2, "FN": 3}

MODEL_TOKENIZERS = {
    "m1": "microsoft/codebert-base",
    "m2": "microsoft/graphcodebert-base",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-file", type=Path, default=ROOT / "results" / "bcb_5k_test_pairs_kaggle.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "error_analysis_diffmul")
    args = parser.parse_args()

    pairs = []
    with open(args.pairs_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"Đã nạp {len(pairs)} cặp từ {args.pairs_file}", flush=True)

    from transformers import AutoTokenizer

    for key, hf_id in MODEL_TOKENIZERS.items():
        scores_path = args.out_dir / f"{key}_all_scores.jsonl"
        rows = []
        with open(scores_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        print(f"[{key}] {len(rows)} dòng điểm từ {scores_path.name}", flush=True)

        tok = AutoTokenizer.from_pretrained(hf_id)
        records: list[dict[str, object]] = []
        for row in rows:
            pair = pairs[row["index"]]
            label, pred = int(row["label"]), int(row["pred"])
            if label == 1 and pred == 1:
                error_type = "TP"
            elif label == 0 and pred == 0:
                error_type = "TN"
            elif label == 0 and pred == 1:
                error_type = "FP"
            else:
                error_type = "FN"
            records.append(
                {
                    "model_label": row["model_label"],
                    "error_type": error_type,
                    "index": 0,  # gán lại sau khi sort theo nhóm
                    "pair_id": row["pair_id"],
                    "id1": row["id1"],
                    "id2": row["id2"],
                    "label": label,
                    "prob": row["prob"],
                    "cosine": row["cosine"],
                    "threshold": row["threshold"],
                    "token1": len(tok.encode(pair["func1"], add_special_tokens=False, truncation=False)),
                    "token2": len(tok.encode(pair["func2"], add_special_tokens=False, truncation=False)),
                }
            )

        # Sort ổn định: TP -> TN -> FP -> FN; giữ nguyên thứ tự gốc trong từng nhóm.
        records.sort(key=lambda rec: GROUP_ORDER[str(rec["error_type"])])
        counters: dict[str, int] = {}
        with (args.out_dir / f"{key}_all_scores.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for rec in records:
                group = str(rec["error_type"])
                counters[group] = counters.get(group, 0) + 1
                rec["index"] = counters[group]
                writer.writerow(rec)
        print(
            f"[{key}] đã ghi {args.out_dir / (key + '_all_scores.csv')} "
            f"({', '.join(f'{g}={counters.get(g, 0)}' for g in GROUP_ORDER)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
