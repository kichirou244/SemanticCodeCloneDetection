"""Chuẩn bị toàn bộ dữ liệu Train (6.000), Val (2.000) và Test (5.000) chuẩn cho Mô hình Lai Ghép Level 3.

Đảm bảo 100% tái tạo chính xác tuyệt đối các mẫu kiểm định và mẫu huấn luyện.
Tự động tải các shard Parquet từ Hugging Face nếu chưa có sẵn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import random
import sys
import urllib3
import requests
import pandas as pd

urllib3.disable_warnings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_hybrid_data")

DATA_DIR = Path("data/raw/bigclonebench")
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEST_SHARDS_DIR = DATA_DIR / "test_shards"
TEST_SHARDS_DIR.mkdir(parents=True, exist_ok=True)
CPG_DIR = Path("artifacts/cpg_export")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_PARQUET = DATA_DIR / "validation_shard_0.parquet"
TRAIN_PARQUET = DATA_DIR / "train_shard_0.parquet"


def download_file(url: str, dest: Path) -> None:
    if not dest.exists() or dest.stat().st_size < 1000:
        logger.info("Đang tải %s từ Hugging Face...", dest.name)
        resp = requests.get(url, verify=False, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info("  -> Đã lưu: %s (%.2f MB)", dest.name, dest.stat().st_size / 1024 / 1024)
    else:
        logger.info("Đã có sẵn: %s (%.2f MB)", dest.name, dest.stat().st_size / 1024 / 1024)


def sample_balanced_train_val(parquet_path: Path, n_total: int, cpg_available: set[str], seed: int = 42) -> list[dict]:
    logger.info("Đang lọc %d mẫu cân bằng từ %s...", n_total, parquet_path.name)
    df = pd.read_parquet(parquet_path)
    half = n_total // 2
    positives, negatives = [], []

    for _, row in df.iterrows():
        id1 = str(row["id1"])
        id2 = str(row["id2"])
        if cpg_available and (id1 not in cpg_available or id2 not in cpg_available):
            continue

        label = 1 if row["label"] else 0
        pair = {
            "id": int(row["id"]),
            "id1": id1,
            "id2": id2,
            "func1": str(row["func1"]).strip(),
            "func2": str(row["func2"]).strip(),
            "label": label,
        }

        if label == 1 and len(positives) < half:
            positives.append(pair)
        elif label == 0 and len(negatives) < half:
            negatives.append(pair)

        if len(positives) >= half and len(negatives) >= half:
            break

    combined = positives + negatives
    rng = random.Random(seed)
    rng.shuffle(combined)
    logger.info("  -> Thu được: %d mẫu (Pos: %d, Neg: %d)", len(combined), len(positives), len(negatives))
    return combined


def build_exact_5k_test_pairs(shards_dir: Path) -> list[dict]:
    logger.info("Đang tái tạo tập 5.000 cặp Test cân bằng chuẩn CodeXGlue...")
    positives, negatives = [], []
    for i in range(3):
        shard = shards_dir / f"test-{i:05d}-of-00003.parquet"
        df = pd.read_parquet(shard)
        for _, item in df.iterrows():
            lbl = 1 if item["label"] else 0
            pair = {
                "id": int(item.get("id")),
                "id1": str(item.get("id1")),
                "id2": str(item.get("id2")),
                "func1": str(item["func1"]).strip(),
                "func2": str(item["func2"]).strip(),
                "label": lbl,
            }
            if lbl == 1:
                positives.append(pair)
            else:
                negatives.append(pair)
            if len(positives) >= 10000 and len(negatives) >= 10000:
                break
        if len(positives) >= 10000 and len(negatives) >= 10000:
            break

    rng = random.Random(42)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    all_pairs = positives[:2500] + negatives[:2500]
    rng.shuffle(all_pairs)
    logger.info("  -> Đã tạo tập 5.000 test chuẩn (Pos: %d, Neg: %d)", sum(1 for p in all_pairs if p["label"] == 1), sum(1 for p in all_pairs if p["label"] == 0))
    return all_pairs


def main():
    # 1. Tải shards Parquet
    download_file(
        "https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/train-00000-of-00006.parquet",
        TRAIN_PARQUET,
    )
    download_file(
        "https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/validation-00000-of-00003.parquet",
        VAL_PARQUET,
    )
    for i in range(3):
        download_file(
            f"https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/test-{i:05d}-of-00003.parquet",
            TEST_SHARDS_DIR / f"test-{i:05d}-of-00003.parquet",
        )

    # 2. Tập hợp danh sách CPG graphs sẵn có
    cpg_available = set()
    if CPG_DIR.exists():
        cpg_available = set(p.stem for p in CPG_DIR.glob("*.json"))
    logger.info("Số đồ thị CPG có sẵn trong %s: %d", CPG_DIR, len(cpg_available))

    # 3. Tạo 6k train, 2k val, 5k test
    out_train = RESULTS_DIR / "bcb_hybrid_train_6k.jsonl"
    out_val = RESULTS_DIR / "bcb_hybrid_val_2k.jsonl"
    out_test = RESULTS_DIR / "bcb_5k_test_pairs.jsonl"

    if not out_train.exists():
        train_pairs = sample_balanced_train_val(TRAIN_PARQUET, n_total=6000, cpg_available=cpg_available, seed=42)
        with open(out_train, "w", encoding="utf-8") as f:
            for p in train_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info("Đã ghi %d mẫu vào %s", len(train_pairs), out_train)

    if not out_val.exists():
        val_pairs = sample_balanced_train_val(VAL_PARQUET, n_total=2000, cpg_available=cpg_available, seed=42)
        with open(out_val, "w", encoding="utf-8") as f:
            for p in val_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info("Đã ghi %d mẫu vào %s", len(val_pairs), out_val)

    if not out_test.exists():
        test_pairs = build_exact_5k_test_pairs(TEST_SHARDS_DIR)
        with open(out_test, "w", encoding="utf-8") as f:
            for p in test_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info("Đã ghi %d mẫu vào %s", len(test_pairs), out_test)

    logger.info("Hoàn tất chuẩn bị toàn bộ 3 tập dữ liệu cho Level 3!")


if __name__ == "__main__":
    main()
