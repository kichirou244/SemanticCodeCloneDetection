from __future__ import annotations

import json
import logging
from pathlib import Path
import random
from typing import Any, Generator, Iterable

import pandas as pd

from src.data.loaders.base import ClonePairSample

logger = logging.getLogger(__name__)


class BigCloneBenchLoader:
    HF_ID: str = "google/code_x_glue_cc_clone_detection_big_clone_bench"
    LANGUAGE: str = "java"
    SPLITS: tuple[str, ...] = ("train", "validation", "test")

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @staticmethod
    def normalize_code(code: str) -> str:
        if not code:
            return ""
        return code.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def coerce_label(raw_label: Any) -> int:
        if isinstance(raw_label, bool):
            return 1 if raw_label else 0
        if isinstance(raw_label, (int, float)):
            return 1 if int(raw_label) > 0 else 0
        if isinstance(raw_label, str):
            val = raw_label.strip().lower()
            if val in ("true", "1", "clone", "yes"):
                return 1
            if val in ("false", "0", "non-clone", "no"):
                return 0
        raise ValueError(f"Khong the nhan dien nhan: {raw_label!r}")

    def parse_row(self, row: dict[str, Any], split: str = "") -> ClonePairSample:
        raw_id = row.get("id")
        pair_id = str(raw_id) if raw_id is not None else f"{row.get('id1')}:::{row.get('id2')}"
        func1 = self.normalize_code(str(row.get("func1", "")))
        func2 = self.normalize_code(str(row.get("func2", "")))
        label = self.coerce_label(row.get("label", 0))

        metadata: dict[str, Any] = {}
        for k in ("clone_type", "type", "similarity"):
            if k in row:
                metadata[k] = row[k]

        return ClonePairSample(
            pair_id=pair_id,
            id1=str(row.get("id1", "")),
            id2=str(row.get("id2", "")),
            func1=func1,
            func2=func2,
            label=label,
            language=self.LANGUAGE,
            split=split,
            metadata=metadata if metadata else None,
        )

    def load_from_hf(
        self, split: str = "train", max_samples: int | None = None, streaming: bool = False
    ) -> list[ClonePairSample]:
        from datasets import load_dataset
        ds = load_dataset(
            self.HF_ID,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            streaming=streaming,
        )
        samples: list[ClonePairSample] = []
        for i, row in enumerate(ds):
            if max_samples is not None and i >= max_samples:
                break
            samples.append(self.parse_row(row, split=split))
        return samples

    def stream_pairs(
        self, split: str = "train", max_samples: int | None = None
    ) -> Generator[ClonePairSample, None, None]:
        from datasets import load_dataset
        ds = load_dataset(
            self.HF_ID,
            split=split,
            cache_dir=str(self.cache_dir) if self.cache_dir else None,
            streaming=True,
        )
        count = 0
        for row in ds:
            if max_samples is not None and count >= max_samples:
                break
            yield self.parse_row(row, split=split)
            count += 1

    @staticmethod
    def sample_subset(
        pairs: list[ClonePairSample], n_samples: int = 500, seed: int = 42, balance: bool = True
    ) -> list[ClonePairSample]:
        if len(pairs) <= n_samples:
            return list(pairs)

        rng = random.Random(seed)
        if not balance:
            indices = list(range(len(pairs)))
            rng.shuffle(indices)
            return [pairs[i] for i in indices[:n_samples]]

        positives = [p for p in pairs if p.label == 1]
        negatives = [p for p in pairs if p.label == 0]
        half = n_samples // 2
        remainder = n_samples - half

        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected_pos = positives[:half]
        selected_neg = negatives[:remainder]

        if len(selected_pos) < half:
            extra = half - len(selected_pos)
            selected_neg.extend(negatives[remainder : remainder + extra])
        elif len(selected_neg) < remainder:
            extra = remainder - len(selected_neg)
            selected_pos.extend(positives[half : half + extra])

        subset = selected_pos + selected_neg
        rng.shuffle(subset)
        return subset

    @staticmethod
    def save_jsonl(pairs: Iterable[ClonePairSample], file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                item = {
                    "pair_id": p.pair_id,
                    "id1": p.id1,
                    "id2": p.id2,
                    "func1": p.func1,
                    "func2": p.func2,
                    "label": p.label,
                    "language": p.language,
                    "split": p.split,
                    "metadata": p.metadata,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, file_path: Path | str, max_samples: int | None = None) -> list[ClonePairSample]:
        path = Path(file_path)
        samples: list[ClonePairSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                samples.append(
                    ClonePairSample(
                        pair_id=str(data.get("pair_id", "")),
                        id1=str(data.get("id1", "")),
                        id2=str(data.get("id2", "")),
                        func1=cls.normalize_code(data.get("func1", "")),
                        func2=cls.normalize_code(data.get("func2", "")),
                        label=cls.coerce_label(data.get("label", 0)),
                        language=data.get("language", cls.LANGUAGE),
                        split=data.get("split", ""),
                        metadata=data.get("metadata"),
                    )
                )
        return samples

    @staticmethod
    def save_parquet(pairs: list[ClonePairSample], file_path: Path | str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "pair_id": p.pair_id,
                "id1": p.id1,
                "id2": p.id2,
                "func1": p.func1,
                "func2": p.func2,
                "label": int(p.label),
                "language": p.language,
                "split": p.split,
                "metadata": json.dumps(p.metadata) if p.metadata else "",
            }
            for p in pairs
        ]
        pd.DataFrame(records).to_parquet(path, index=False)

    @classmethod
    def load_parquet(cls, file_path: Path | str, max_samples: int | None = None) -> list[ClonePairSample]:
        path = Path(file_path)
        df = pd.read_parquet(path)
        if max_samples is not None:
            df = df.iloc[:max_samples]

        samples: list[ClonePairSample] = []
        for _, row in df.iterrows():
            raw_meta = row.get("metadata", "")
            meta = json.loads(raw_meta) if raw_meta and isinstance(raw_meta, str) else None
            samples.append(
                ClonePairSample(
                    pair_id=str(row["pair_id"]),
                    id1=str(row["id1"]),
                    id2=str(row["id2"]),
                    func1=cls.normalize_code(str(row["func1"])),
                    func2=cls.normalize_code(str(row["func2"])),
                    label=int(row["label"]),
                    language=str(row.get("language", cls.LANGUAGE)),
                    split=str(row.get("split", "")),
                    metadata=meta,
                )
            )
        return samples

    def describe_schema(self) -> dict[str, Any]:
        return {
            "hf_id": self.HF_ID,
            "language": self.LANGUAGE,
            "splits": list(self.SPLITS),
            "expected_fields": {
                "id": "ID cua cap",
                "id1": "ID ham thu 1",
                "id2": "ID ham thu 2",
                "func1": "Ma nguon Java ham 1",
                "func2": "Ma nguon Java ham 2",
                "label": "1 neu la Clone, 0 neu khong",
            },
        }
