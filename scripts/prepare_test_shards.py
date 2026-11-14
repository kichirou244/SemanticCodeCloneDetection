"""Tải các test shard parquet từ Hugging Face và lập bản đồ id -> (id1, id2)."""

from pathlib import Path
import json
import urllib3
import requests
import pandas as pd

urllib3.disable_warnings()

OUT_DIR = Path("data/raw/bigclonebench/test_shards")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    f"https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench/resolve/main/data/test-{i:05d}-of-00003.parquet"
    for i in range(3)
]

def download_shards():
    for i, url in enumerate(URLS):
        dest = OUT_DIR / f"test-{i:05d}-of-00003.parquet"
        if not dest.exists() or dest.stat().st_size < 1000:
            print(f"Đang tải {dest.name} từ Hugging Face...", flush=True)
            resp = requests.get(url, verify=False, stream=True)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print(f"  -> Hoàn tất: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
        else:
            print(f"Đã có sẵn: {dest.name} ({dest.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)

def build_id_map() -> dict[int, tuple[str, str]]:
    print("Đang nạp và lập chỉ mục id -> (id1, id2)...", flush=True)
    id_map = {}
    for i in range(3):
        shard = OUT_DIR / f"test-{i:05d}-of-00003.parquet"
        df = pd.read_parquet(shard, columns=["id", "id1", "id2"])
        for _, row in df.iterrows():
            id_map[int(row["id"])] = (str(row["id1"]), str(row["id2"]))
    print(f"Tổng số cặp test trong CodeXGlue: {len(id_map):,}", flush=True)
    return id_map

if __name__ == "__main__":
    download_shards()
    id_map = build_id_map()
    
    # Kiểm tra khớp với results/bcb_5k_test_pairs.jsonl
    test_5k_file = Path("results/bcb_5k_test_pairs.jsonl")
    enriched_pairs = []
    cpg_dir = Path("artifacts/cpg_export")
    missing_cpg_count = 0
    with open(test_5k_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pair_id = int(item["id"])
            if pair_id in id_map:
                id1, id2 = id_map[pair_id]
                item["id1"] = id1
                item["id2"] = id2
                if not (cpg_dir / f"{id1}.json").exists():
                    missing_cpg_count += 1
                if not (cpg_dir / f"{id2}.json").exists():
                    missing_cpg_count += 1
            enriched_pairs.append(item)
            
    print(f"Đã làm giàu {len(enriched_pairs)} mẫu test!")
    print(f"Số đồ thị CPG bị thiếu: {missing_cpg_count} / {len(enriched_pairs) * 2}")
    
    enriched_file = Path("results/bcb_5k_test_pairs_enriched.jsonl")
    with open(enriched_file, "w", encoding="utf-8") as f:
        for p in enriched_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Đã lưu: {enriched_file} ({enriched_file.stat().st_size / 1024 / 1024:.2f} MB)")
