"""Truy xuất samples/pairs kiểu streaming, không ôm code vào RAM (§4-§5).

- TF-IDF fit ở sample level: stream code từng dòng từ processed jsonl.
- Sau vectorize: giữ `id → row index` + sparse matrix (indexed vectors).
- Pairs: stream theo batch, lookup vector theo id, không duplicate strings.
"""

import json
from pathlib import Path


def stream_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def stream_codes(processed_path: str | Path, split: str):
    """Yield (sample_id, code) của một split, đọc streaming từng dòng."""
    for row in stream_jsonl(processed_path):
        if row["split"] == split:
            yield row["sample_id"], row["code"]


def stream_pairs(pairs_path: str | Path, batch_size: int = 4096):
    """Yield từng batch pair dicts từ file jsonl."""
    batch = []
    for row in stream_jsonl(pairs_path):
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
