"""Unit tests cho module audit và leakage cặp dữ liệu (pair_audit.py)."""

import pytest

from src.data.loaders.base import ClonePairSample
from src.data.validators.pair_audit import audit_pairs, check_pair_leakage, compute_code_hash


@pytest.fixture
def clean_splits() -> dict[str, list[ClonePairSample]]:
    train = [
        ClonePairSample(
            pair_id="t1",
            id1="1",
            id2="2",
            func1="public void a() { int x = 1; }",
            func2="public void b() { int y = 1; }",
            label=1,
            split="train",
        ),
        ClonePairSample(
            pair_id="t2",
            id1="3",
            id2="4",
            func1="public void c() { return; }",
            func2="public void d() { System.out.println(); }",
            label=0,
            split="train",
        ),
    ]
    val = [
        ClonePairSample(
            pair_id="v1",
            id1="5",
            id2="6",
            func1="public int e() { return 42; }",
            func2="public int f() { return 42; }",
            label=1,
            split="val",
        )
    ]
    return {"train": train, "val": val}


def test_audit_pairs_empty():
    res = audit_pairs([])
    assert res["n_pairs"] == 0
    assert res["positive_ratio"] == 0.0


def test_audit_pairs_basic():
    pairs = [
        ClonePairSample(
            pair_id="p1",
            id1="1",
            id2="2",
            func1="line1\nline2\nline3",
            func2="short",
            label=1,
        ),
        ClonePairSample(
            pair_id="p2",
            id1="3",
            id2="4",
            func1="hello world",
            func2="another code",
            label=0,
        ),
    ]
    res = audit_pairs(pairs)
    assert res["n_pairs"] == 2
    assert res["n_positive"] == 1
    assert res["n_negative"] == 1
    assert res["positive_ratio"] == 0.5
    assert res["unique_func_ids"] == 4
    assert res["missing_code"] == 0
    assert res["duplicate_pairs"] == 0
    assert res["conflicting_pairs"] == 0
    assert res["length_stats"]["lines_max"] == 3


def test_audit_pairs_conflict():
    pairs = [
        ClonePairSample(pair_id="p1", id1="10", id2="20", func1="a", func2="b", label=1),
        ClonePairSample(pair_id="p2", id1="20", id2="10", func1="b", func2="a", label=0),  # đảo id1/id2
    ]
    res = audit_pairs(pairs)
    assert res["duplicate_pairs"] == 1
    assert res["conflicting_pairs"] == 1


def test_compute_code_hash():
    c1 = "public   void test()  {\n\n}"
    c2 = "public void test() {}"
    assert compute_code_hash(c1) == compute_code_hash(c2)


def test_check_pair_leakage_clean(clean_splits):
    result = check_pair_leakage(clean_splits)
    assert not result["has_critical_pair_leakage"]
    assert result["pair_overlaps"]["train_vs_val"] == 0


def test_check_pair_leakage_detected(clean_splits):
    # Tạo cặp bị leak vào val (ngay cả khi đảo id1 và id2)
    leaked_val = clean_splits["val"] + [
        ClonePairSample(
            pair_id="v_leak",
            id1="2",
            id2="1",
            func1="code",
            func2="code",
            label=1,
            split="val",
        )
    ]
    dirty_splits = {"train": clean_splits["train"], "val": leaked_val}
    result = check_pair_leakage(dirty_splits)
    assert result["has_critical_pair_leakage"]
    assert result["pair_overlaps"]["train_vs_val"] == 1
