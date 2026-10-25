"""Bat bien cua manifest custom_suite sau khi mo rong non-clone.

Khong can GPU/model: chi doc `data/custom_suite/pairs.jsonl`.
"""

from __future__ import annotations

from pathlib import Path

from scripts.build_custom_suite_negatives import _read_pairs, build_new_rows, validate

ROOT = Path(__file__).resolve().parents[1]


def _group(filename: str) -> str:
    return Path(filename).stem.split("_", 1)[0]


def test_manifest_counts_and_no_duplicate() -> None:
    rows = _read_pairs()
    validate(rows)  # file ton tai, khong cap trung, pair_id duy nhat
    clone = [r for r in rows if int(r["label"]) == 1]
    non = [r for r in rows if int(r["label"]) == 0]
    assert len(rows) == 89
    assert len(clone) == 25
    assert len(non) == 64
    # Non-clone phai nhieu hon clone (gan voi thuc te retrieval)
    assert len(non) > 2 * len(clone)


def test_labels_follow_problem_boundary() -> None:
    for r in _read_pairs():
        same_problem = _group(r["a"]) == _group(r["b"])
        if same_problem:
            # Trong cung bai, chi cac cap da duoc gan nhan thu cong moi ton tai.
            # Moi cap non-clone cung bai phai co dung mot file n_.
            if int(r["label"]) == 0:
                assert ("_n_" in r["a"]) != ("_n_" in r["b"]), r["pair_id"]
        else:
            # Khac bai => khong the clone.
            assert int(r["label"]) == 0, r["pair_id"]


def test_generator_is_idempotent() -> None:
    rows = _read_pairs()
    assert build_new_rows(rows) == [], "chay lai generator khong duoc them cap moi"
