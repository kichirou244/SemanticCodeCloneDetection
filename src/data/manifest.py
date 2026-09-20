"""DatasetManifest — mô tả đầy đủ để tái tạo dataset (Step 20)."""

import json
from pathlib import Path


def build_manifest(
    dataset: str,
    language: str,
    seed: int,
    split_strategy: str,
    cleaning: dict,
    dedup: dict,
    splits: dict,
    pairs: dict,
    artifacts: dict,
    config_snapshot: str = "",
) -> dict:
    return {
        "dataset": dataset,
        "language": language,
        "version": "phase1-v1",
        "seed": seed,
        "split_strategy": split_strategy,
        "cleaning": cleaning,
        "deduplication": dedup,
        "splits": splits,
        "pairs": pairs,
        "pair_ratio": "1:1",
        "artifacts": artifacts,
        "config_snapshot": config_snapshot,
    }


def save_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
