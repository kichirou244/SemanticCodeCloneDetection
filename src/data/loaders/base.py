"""Interface thống nhất cho mọi dataset (Step 8 — Unified Interfaces)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CodeSample:
    """Một mẫu mã nguồn chuẩn hóa xuyên dataset."""

    sample_id: str
    code: str  # raw source — không bao giờ bị ghi đè bởi normalization
    problem_id: str  # problem/task identity — dùng để split chống leakage
    language: str  # "cpp" | "python" | "java"
    split: str = ""  # train | validation | test
    normalized_code: str = ""  # chỉ dùng cho dedup/leakage, không thay raw
    metadata: dict | None = None


@dataclass
class ClonePairSample:
    """Một cặp mã nguồn đối sánh trong bài toán phát hiện clone (như BigCloneBench)."""

    pair_id: str
    id1: str
    id2: str
    func1: str
    func2: str
    label: int  # 1: clone, 0: non-clone
    language: str = "java"
    split: str = ""  # train | validation | test
    metadata: dict | None = None


class DatasetAdapter(ABC):
    """Mọi loader (POJ-104, CodeNet Python) phải implement interface này."""

    language: str = ""

    @abstractmethod
    def load(self, **kwargs) -> list[CodeSample]:
        """Nạp + chuẩn hóa về list[CodeSample]. Chưa split/augment ở đây."""
        raise NotImplementedError

    @abstractmethod
    def describe_schema(self) -> dict:
        """Mô tả schema gốc của dataset (phục vụ audit)."""
        raise NotImplementedError
