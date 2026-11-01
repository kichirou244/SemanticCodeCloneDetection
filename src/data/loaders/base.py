from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CodeSample:
    sample_id: str
    code: str
    problem_id: str
    language: str
    split: str = ""
    normalized_code: str = ""
    metadata: dict | None = None


@dataclass
class ClonePairSample:
    pair_id: str
    id1: str
    id2: str
    func1: str
    func2: str
    label: int
    language: str = "java"
    split: str = ""
    metadata: dict | None = None


class DatasetAdapter(ABC):
    language: str = ""

    @abstractmethod
    def load(self, **kwargs) -> list[CodeSample]:
        raise NotImplementedError

    @abstractmethod
    def describe_schema(self) -> dict:
        raise NotImplementedError
