from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ClonePair:
    code_a: str
    code_b: str
    label: int
    problem_a: str = ""
    problem_b: str = ""


class CodeRepresentation(ABC):
    @abstractmethod
    def encode(self, codes: list[str]):
        raise NotImplementedError


class CloneDetector(ABC):
    @abstractmethod
    def predict_proba(self, pairs: list[ClonePair]) -> list[float]:
        raise NotImplementedError

    def predict(self, pairs: list[ClonePair], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(pairs)]


class Experiment(ABC):
    @abstractmethod
    def run(self, config: dict) -> dict:
        raise NotImplementedError
