"""Test leakage pair-level: sample/code/pair overlap xuyên split."""

from src.data.leakage.checks import (
    canonical_pair,
    check_pair_overlap,
    check_sample_overlap,
    run_pair_leakage_audit,
)
from src.data.loaders.base import CodeSample


def _s(sid, code, prob):
    return CodeSample(sid, code, prob, "cpp")


def test_pair_overlap_counts_reversed_as_same():
    pairs = {
        "train": [{"code_a_id": "a", "code_b_id": "b"}],
        "test": [{"code_a_id": "b", "code_b_id": "a"}],
    }
    r = check_pair_overlap(pairs)
    assert r["pair_overlap"]["train x test"] == 1 and r["ok"] is False


def test_full_pair_audit_pass_and_fail():
    good = {
        "train": [_s("a", "code one here", "p1")],
        "test": [_s("b", "code two here!", "p2")],
    }
    gpairs = {"train": [{"code_a_id": "a", "code_b_id": "a2"}],
              "test": [{"code_a_id": "b", "code_b_id": "b2"}]}
    assert run_pair_leakage_audit(good, gpairs)["passed"] is True

    bad = {
        "train": [_s("a", "same code here!", "p1")],
        "test": [_s("b", "same code here!", "p2")],
    }
    bpairs = {"train": [{"code_a_id": "a", "code_b_id": "x"}],
              "test": [{"code_a_id": "a", "code_b_id": "y"}]}
    r = run_pair_leakage_audit(bad, bpairs)
    assert r["passed"] is False
    assert r["code"]["ok"] is False  # cùng code, khác sample_id → code overlap bắt được
    assert r["normalized_code"]["ok"] is False
    assert canonical_pair("a", "b") == "a:::b"


def test_sample_overlap_same_id_across_splits():
    splits = {
        "train": [_s("a", "code one here yes", "p1")],
        "test": [_s("a", "code one here yes", "p1")],
    }
    r = run_pair_leakage_audit(splits, {"train": [], "test": []})
    assert r["sample"]["sample_overlap"]["train x test"] == 1
    assert r["sample"]["ok"] is False
    assert r["passed"] is False
