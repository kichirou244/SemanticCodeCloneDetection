"""Test split theo problem: test không chứa problem của train."""

from src.data.loaders.base import CodeSample
from src.data.splitting.problem_split import split_by_problem


def _samples():
    out = []
    for p in range(10):
        for i in range(3):
            out.append(CodeSample(f"{p}-{i}", "code", f"prob{p}", "cpp"))
    return out


def test_no_problem_overlap():
    splits = split_by_problem(_samples(), seed=42)
    assert not (set(s.problem_id for s in splits["train"]) & set(s.problem_id for s in splits["test"]))
