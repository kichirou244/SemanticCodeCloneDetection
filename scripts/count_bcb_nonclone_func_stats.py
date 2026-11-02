#!/usr/bin/env python3
"""
Thong ke so luong cap khong trung lap cung/khac ham tren BigCloneBench (CodeXGLUE)
Dung streaming=True cua datasets de khong ton dung luong o cung (~5.5GB).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "google/code_x_glue_cc_clone_detection_big_clone_bench"
SPLITS = ["test", "validation", "train"]
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "results" / "bcb_negative_pairs_func_stats.json"

FUNC_PATTERN = re.compile(
    r'(?:public|protected|private|static|final|synchronized|\s)+[\w<>\[\],\s]+\s+(\w+)\s*\(',
    re.MULTILINE
)
FALLBACK_PATTERN = re.compile(r'(\w+)\s*\(', re.MULTILINE)
KEYWORD_IGNORE = {'if', 'for', 'while', 'switch', 'catch', 'synchronized', 'return'}


def extract_func_name(code: str) -> str:
    lines = [line for line in code.split('\n') if not line.strip().startswith('@')]
    clean_code = '\n'.join(lines)

    match = FUNC_PATTERN.search(clean_code)
    if match:
        name = match.group(1)
        if name not in KEYWORD_IGNORE:
            return name

    match_fallback = FALLBACK_PATTERN.search(clean_code)
    if match_fallback:
        name = match_fallback.group(1)
        if name not in KEYWORD_IGNORE:
            return name

    return "UNKNOWN"


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_stats = {}

    print(f"Bat dau quet 3 tap (test, validation, train) tu dataset {DATASET_NAME} qua streaming...")
    start_total = time.time()

    for split in SPLITS:
        t0 = time.time()
        print(f"\n[{split.upper()}] Dang ket noi va quet stream...")
        ds = load_dataset(DATASET_NAME, split=split, streaming=True)

        total_pairs = 0
        pos_pairs = 0
        neg_pairs = 0
        neg_same_name = 0
        neg_diff_name = 0
        same_name_examples = []
        diff_name_examples = []

        for item in tqdm(ds, desc=f"Quet {split}"):
            total_pairs += 1
            is_clone = bool(item["label"])

            if not is_clone:
                neg_pairs += 1
                name1 = extract_func_name(item["func1"])
                name2 = extract_func_name(item["func2"])

                if name1 != "UNKNOWN" and name1 == name2:
                    neg_same_name += 1
                    if len(same_name_examples) < 5:
                        same_name_examples.append({
                            "id": item.get("id"),
                            "method1": name1,
                            "method2": name2,
                        })
                else:
                    neg_diff_name += 1
                    if len(diff_name_examples) < 5:
                        diff_name_examples.append({
                            "id": item.get("id"),
                            "method1": name1,
                            "method2": name2,
                        })
            else:
                pos_pairs += 1

        elapsed = time.time() - t0
        pct_same = (neg_same_name / neg_pairs * 100) if neg_pairs > 0 else 0.0
        pct_diff = (neg_diff_name / neg_pairs * 100) if neg_pairs > 0 else 0.0

        split_result = {
            "total_pairs": total_pairs,
            "positives_clone": pos_pairs,
            "negatives_nonclone": neg_pairs,
            "negatives_same_func_name": neg_same_name,
            "negatives_diff_func_name": neg_diff_name,
            "pct_same_name": round(pct_same, 4),
            "pct_diff_name": round(pct_diff, 4),
            "time_seconds": round(elapsed, 2),
            "sample_same_name": same_name_examples,
            "sample_diff_name": diff_name_examples,
        }
        all_stats[split] = split_result

        print(f"[{split.upper()}] Hoan tat trong {elapsed:.1f}s:")
        print(f"  - Tong: {total_pairs:,} | Non-clones (label=0): {neg_pairs:,}")
        print(f"  - Cung ten ham: {neg_same_name:,} ({pct_same:.2f}%)")
        print(f"  - Khac ten ham: {neg_diff_name:,} ({pct_diff:.2f}%)")

        # Ghi ngay tung split vao file ket qua de luu tien do
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - start_total
    all_stats["_total_time_seconds"] = round(total_elapsed, 2)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\nDa ghi toan bo ket qua vao: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
