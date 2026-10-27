"""BigCloneBench Loader (Java) — CodeXGLUE Code-to-Code Clone Detection.

Nguồn: https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_big_clone_bench
Bộ dữ liệu gồm các cặp hàm Java (methods) trích xuất từ IJaDataset-2.0 và được gán nhãn
bởi BigCloneBench benchmark:
  - id: id của cặp mẫu (int/str)
  - id1, id2: id của hàm thứ 1 và hàm thứ 2
  - func1, func2: mã nguồn Java của hàm thứ 1 và hàm thứ 2
  - label: True/1 (clone) hoặc False/0 (non-clone)
"""

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
    """Loader cho BigCloneBench Java dataset từ CodeXGLUE."""

    HF_ID: str = "google/code_x_glue_cc_clone_detection_big_clone_bench"
    LANGUAGE: str = "java"
    SPLITS: tuple[str, ...] = ("train", "validation", "test")

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @staticmethod
    def normalize_code(code: str) -> str:
        """Chuẩn hóa ký tự xuống dòng và khoảng trắng cơ bản."""
        if not code:
            return ""
        # Chuẩn hóa CRLF thành LF và strip khoảng trắng thừa ở hai đầu
        return code.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def coerce_label(raw_label: Any) -> int:
        """Chuyển đổi các định dạng label khác nhau (bool, int, str) thành 0 hoặc 1."""
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
        raise ValueError(f"Không thể nhận diện nhãn: {raw_label!r}")

    def parse_row(self, row: dict[str, Any], split: str = "") -> ClonePairSample:
        """Chuyển một dict thô từ dataset thành một đối tượng ClonePairSample chuẩn hóa."""
        raw_id = row.get("id")
        pair_id = str(raw_id) if raw_id is not None else f"{row.get('id1')}:::{row.get('id2')}"
        id1 = str(row.get("id1", ""))
        id2 = str(row.get("id2", ""))
        func1 = self.normalize_code(str(row.get("func1", "")))
        func2 = self.normalize_code(str(row.get("func2", "")))
        label = self.coerce_label(row.get("label", 0))

        metadata: dict[str, Any] = {}
        for k in ("clone_type", "type", "similarity"):
            if k in row:
                metadata[k] = row[k]

        return ClonePairSample(
            pair_id=pair_id,
            id1=id1,
            id2=id2,
            func1=func1,
            func2=func2,
            label=label,
            language=self.LANGUAGE,
            split=split,
            metadata=metadata if metadata else None,
        )

    def load_from_hf(
        self,
        split: str = "train",
        max_samples: int | None = None,
        streaming: bool = False,
    ) -> list[ClonePairSample]:
        """Tải dữ liệu trực tiếp từ Hugging Face qua thư viện datasets."""
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
        self,
        split: str = "train",
        max_samples: int | None = None,
    ) -> Generator[ClonePairSample, None, None]:
        """Stream các cặp dữ liệu từ Hugging Face mà không tải toàn bộ vào RAM."""
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
        pairs: list[ClonePairSample],
        n_samples: int = 500,
        seed: int = 42,
        balance: bool = True,
    ) -> list[ClonePairSample]:
        """Trích xuất tập con có kiểm soát seed cố định và cân bằng nhãn 1:1 nếu yêu cầu."""
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

        # Nếu một bên không đủ, bù bằng bên còn lại
        if len(selected_pos) < half:
            extra_needed = half - len(selected_pos)
            selected_neg.extend(negatives[remainder : remainder + extra_needed])
        elif len(selected_neg) < remainder:
            extra_needed = remainder - len(selected_neg)
            selected_pos.extend(positives[half : half + extra_needed])

        subset = selected_pos + selected_neg
        rng.shuffle(subset)
        return subset

    @staticmethod
    def save_jsonl(pairs: Iterable[ClonePairSample], file_path: Path | str) -> None:
        """Lưu danh sách cặp vào tệp JSON Lines."""
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
        """Nạp danh sách cặp từ tệp JSON Lines."""
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
        """Lưu danh sách cặp vào file Parquet chuẩn hóa (tối ưu nạp và dung lượng)."""
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
        df = pd.DataFrame(records)
        df.to_parquet(path, index=False)

    @classmethod
    def load_parquet(cls, file_path: Path | str, max_samples: int | None = None) -> list[ClonePairSample]:
        """Nạp danh sách cặp từ file Parquet."""
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
        """Mô tả lược đồ dữ liệu và thông tin benchmark."""
        return {
            "hf_id": self.HF_ID,
            "language": self.LANGUAGE,
            "splits": list(self.SPLITS),
            "expected_fields": {
                "id": "int32/str - ID của cặp",
                "id1": "int32/str - ID của hàm thứ nhất",
                "id2": "int32/str - ID của hàm thứ hai",
                "func1": "string - Mã nguồn Java của hàm thứ nhất",
                "func2": "string - Mã nguồn Java của hàm thứ hai",
                "label": "bool/int - 1 nếu là Clone, 0 nếu là Non-clone",
            },
            "subsets": {
                "smoke": 500,
                "dev": 20000,
                "full_train": 901028,
                "full_val": 415416,
                "full_test": 415416,
            },
            "clone_types": [
                "Type-1 (Exact)",
                "Type-2 (Renamed)",
                "Strongly Type-3 (ST3 >= 70% syntactical similarity)",
                "Moderately Type-3 (MT3 50-70% similarity)",
                "Weakly Type-3 / Type-4 (< 50% syntactical similarity, semantic clones)",
            ],
        }
