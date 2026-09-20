"""Smoke test B1 trên subset nhỏ (không cần GPU, không tải model nặng)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.seed import set_global_seed
from src.core.interfaces import ClonePair
from src.evaluation.metrics import classification_metrics
from src.models.baseline.tfidf import TfidfCloneDetector

CODES = [
    "int add(int a, int b) { return a + b; }",
    "int sum(int x, int y) { return x + y; }",
    "void print_hello() { printf(\"hello\"); }",
    "def add(a, b):\n    return a + b",
]


def main() -> None:
    set_global_seed(42)
    det = TfidfCloneDetector(max_features=1000)
    det.fit(CODES)
    pairs = [
        ClonePair(CODES[0], CODES[1], 1),  # cùng ngữ nghĩa, khác tên
        ClonePair(CODES[0], CODES[2], 0),  # khác ngữ nghĩa
        ClonePair(CODES[0], CODES[3], 1),  # cùng ngữ nghĩa, khác ngôn ngữ
    ]
    scores = det.predict_proba(pairs)
    print("scores:", [round(s, 4) for s in scores])
    print(classification_metrics([p.label for p in pairs], scores))


if __name__ == "__main__":
    main()
