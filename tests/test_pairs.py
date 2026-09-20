"""Test pair generation: đúng pos/neg, không trùng pair, A/B=B/A, deterministic."""

from src.data.loaders.base import CodeSample
from src.data.pairs.generator import PairGenConfig, PairGenerator


def _samples():
    out = []
    for p in ["p1", "p2", "p3"]:
        for i in range(6):
            out.append(CodeSample(f"{p}-{i}", f"code {p} {i}", p, "cpp"))
    return out


def test_pos_neg_correctness_and_ratio():
    gen = PairGenerator(PairGenConfig(seed=7, negative_ratio=1))
    pairs = gen.generate(_samples(), "train")
    pos = [p for p in pairs if p.label == 1]
    neg = [p for p in pairs if p.label == 0]
    assert len(pos) == len(neg)  # 1:1
    assert all(p.problem_a == p.problem_b for p in pos)
    assert all(p.problem_a != p.problem_b for p in neg)
    assert all(p.code_a_id != p.code_b_id for p in pairs)


def test_no_duplicate_unordered_pairs():
    gen = PairGenerator(PairGenConfig(seed=7))
    pairs = gen.generate(_samples(), "train")
    keys = set()
    for p in pairs:
        key = PairGenerator.canonical_pair_id(p.code_a_id, p.code_b_id)
        assert key not in keys
        keys.add(key)
    assert PairGenerator.canonical_pair_id("x", "y") == PairGenerator.canonical_pair_id("y", "x")


def test_deterministic_and_cap():
    cfg = PairGenConfig(seed=7, max_positive_pairs_per_problem=5)
    a = [(p.pair_id, p.code_a_id, p.code_b_id) for p in PairGenerator(cfg).generate(_samples(), "train")]
    b = [(p.pair_id, p.code_a_id, p.code_b_id) for p in PairGenerator(cfg).generate(_samples(), "train")]
    assert a == b
    assert sum(1 for p in PairGenerator(cfg).generate(_samples(), "train") if p.label == 1) == 15
