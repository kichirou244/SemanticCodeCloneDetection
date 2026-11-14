"""Chuẩn bị dữ liệu huấn luyện và kiểm định cân bằng cho mô hình lai ghép M5."""

from pathlib import Path
import json
import urllib3
import requests
import pandas as pd
import random

urllib3.disable_warnings()

DATA_DIR = Path("data/raw/bigclonebench")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CPG_DIR = Path("artifacts/cpg_export")

VAL_PARQUET = DATA_DIR / "validation_shard_0.parquet"
TRAIN_PARQUET = DATA_DIR / "train_shard_0.parquet"

def download_file(url: str, dest: Path):
    if not dest.exists() or dest.stat().st_size < 1000:
        print(f"Đang tải {dest.name}...", flush=True)
        resp = requests.get(url, verify=False, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print(f"  -> Đã lưu: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    else:
        print(f"Đã có: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

def sample_balanced_pairs(parquet_path: Path, n_total: int, seed: int = 42) -> list[dict]:
    print(f"Đang nạp và lọc {n_total} mẫu cân bằng từ {parquet_path.name}...", flush=True)
    df = pd.read_parquet(parquet_path)
    
    half = n_total // 2
    positives, negatives = [], []
    
    for _, row in df.iterrows():
        id1 = str(row["id1"])
        id2 = str(row["id2"])
        # Kiểm tra đồ thị CPG có sẵn
        if not (CPG_DIR / f"{id1}.json").exists() or not (CPG_DIR / f"{id2}.json").exists():
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
    print(f"  -> Thu được: {len(combined)} mẫu (Pos: {len(positives)}, Neg: {len(negatives)})", flush=True)
    return combined

def main():
    download_file(
        "https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/validation-00000-of-00003.parquet",
        VAL_PARQUET,
    )
    download_file(
        "https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/train-00000-of-00006.parquet",
        TRAIN_PARQUET,
    )
    
    # Sinh tập train 6k và val 2k
    train_pairs = sample_balanced_pairs(TRAIN_PARQUET, n_total=6000, seed=42)
    val_pairs = sample_balanced_pairs(VAL_PARQUET, n_total=2000, seed=42)
    
    out_train = Path("results/bcb_hybrid_train_6k.jsonl")
    out_val = Path("results/bcb_hybrid_val_2k.jsonl")
    
    with open(out_train, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Đã lưu: {out_train} ({out_train.stat().st_size / 1024 / 1024:.2f} MB)")
    
    with open(out_val, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Đã lưu: {out_val} ({out_val.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()
