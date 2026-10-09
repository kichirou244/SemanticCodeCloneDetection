"""Cặp tương ứng do model bắt: mutual-best, B1 k-gram, map sang token."""

from __future__ import annotations

import torch

from src.demo.alignment.lexer import lex_code
from src.demo.correspondence import mutual_best, unit_rows, word_vectors
from src.demo.service import _attach_token_indices


def test_mutual_best_diagonal():
    v = unit_rows(torch.eye(4))
    got = mutual_best(v, v)
    assert [(i, j) for i, j, _ in got] == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_mutual_best_empty_and_cap():
    assert mutual_best(torch.zeros((0, 8)), torch.zeros((0, 8))) == []
    v = unit_rows(torch.eye(10))
    assert len(mutual_best(v, v, cap=3)) == 3


def test_mutual_best_ignores_nonpositive():
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[-1.0, 0.0]])
    assert mutual_best(unit_rows(a), unit_rows(b)) == []


def test_word_vectors_groups_subwords():
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    vecs, spans = word_vectors(h, [0, 0, 1], [(0, 3), (3, 6), (7, 9)])
    assert spans == [[0, 6], [7, 9]]
    assert vecs.shape == (2, 2)


def test_word_vectors_skips_specials():
    h = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    vecs, spans = word_vectors(h, [None, 0], [(0, 0), (5, 6)])
    assert spans == [[5, 6]] and vecs.shape == (2 - 1, 2)


def test_moss_pairs_identical_deterministic():
    from pathlib import Path

    from src.demo.scorers.moss_scorer import MossScorer

    code = "int main() { int a = 1; return a; }"
    sc = MossScorer("B1", "moss", Path("."), k=20, w=5)
    e1 = sc._evidence_local(code, code)
    e2 = sc._evidence_local(code, code)
    assert e1["pairs"] and e1["pairs"] == e2["pairs"]
    for p in e1["pairs"]:
        assert p["a"] and p["b"]


def test_moss_pairs_unrelated_empty():
    from pathlib import Path

    from src.demo.scorers.moss_scorer import MossScorer

    sc = MossScorer("B1", "moss", Path("."), k=20, w=5)
    e = sc._evidence_local("int main() { int a = 1; }",
                           "string xyz = 'qqq wwwww eeeee'; " * 10)
    assert e["spans_a"] == [] and e.get("pairs", []) == []


def test_service_maps_pairs_to_tokens():
    code_a = "int f() { return 1; }\n"
    code_b = "int f() { return 1; }\n"
    lines_a, lines_b = lex_code(code_a), lex_code(code_b)
    ev = {"kind": "kgram", "note": "x", "n_shared": 1,
          "spans_a": [], "spans_b": [],
          "pairs": [{"a": [0, 5], "b": [0, 5]}]}
    out = _attach_token_indices(ev, lines_a, lines_b, code_a, code_b)
    assert len(out["tok_pairs"]) == 1
    assert out["tok_pairs"][0]["a"] and out["tok_pairs"][0]["b"]
