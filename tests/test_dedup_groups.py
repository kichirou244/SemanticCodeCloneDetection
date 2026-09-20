"""Test dedup: exact/normalized/cross-problem groups + canonical deterministic."""

from src.data.deduplication.groups import DuplicateDetector
from src.data.loaders.base import CodeSample
from src.data.normalization.normalizer import CodeNormalizer


def _s(sid, code, prob="p1"):
    s = CodeSample(sid, code, prob, "cpp")
    s.normalized_code = CodeNormalizer().normalize(code)
    return s


def test_exact_and_cross_problem():
    samples = [
        _s("b", "int x(){return 1;}", "p2"),
        _s("a", "int x(){return 1;}", "p1"),  # trùng exact, khác problem
        _s("c", "int y(){return 2;}", "p1"),
    ]
    res = DuplicateDetector().detect(samples)
    assert res.summary()["n_removed"] == 1
    assert res.dup_to_canonical == {"b": "a"}  # canonical = id nhỏ nhất
    assert [s.sample_id for s in res.canonical] == ["a", "c"]


def test_normalized_only():
    samples = [
        _s("a", "int x(){return 1;}\n"),
        _s("b", "int x(){return 1;}  \n\n"),  # khác raw, cùng normalized
    ]
    res = DuplicateDetector().detect(samples)
    assert res.summary()["n_groups_normalized"] == 1
    assert res.dup_to_canonical == {"b": "a"}


def test_no_false_grouping():
    samples = [_s("a", "int x(){return 1;}"), _s("b", "int x(){return 2;}")]
    res = DuplicateDetector().detect(samples)
    assert res.summary()["n_removed"] == 0
