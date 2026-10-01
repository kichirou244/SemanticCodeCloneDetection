"""Unit test cho bộ căn chỉnh demo — trọng tâm là màu khớp/lệch đúng nguyên tắc.

Không cần model: chỉ dùng `lexer` + `aligner`.
"""

from __future__ import annotations

from src.demo.alignment.aligner import align
from src.demo.alignment.lexer import lex_code


def payload(a: str, b: str, **kwargs) -> dict:
    """Dựng payload render cho một cặp code."""
    lines_a, lines_b = lex_code(a), lex_code(b)
    return align(lines_a, lines_b, **kwargs).to_payload(lines_a, lines_b)


PAIR_RENAME = (
    "int sumArray(int arr[], int n) { int total = 0; return total; }",
    "int accumulate(int values[], int count) { int acc = 0; return acc; }",
)

PAIR_UNRELATED = (
    "long factorial(int n) { return n; }",
    "int countWords(const char *s) { return 0; }",
)


def test_identical_code_matches_everything():
    code = "int max(int a, int b) { return a > b ? a : b; }"
    result = align(lex_code(code), lex_code(code), mode="abstract").to_payload(
        lex_code(code), lex_code(code)
    )
    assert result["stats"]["n_blocks"] >= 1
    assert result["stats"]["coverage_a"] == 1.0
    assert result["stats"]["coverage_b"] == 1.0


def test_abstract_mode_hides_renamed_identifiers():
    result = payload(*PAIR_RENAME, mode="abstract")
    # dòng đầu tiên của A: int → type, sumArray → ID, ... → hầu hết khớp
    assert result["stats"]["coverage_a"] >= 0.9
    assert result["stats"]["coverage_b"] >= 0.9
    assert result["stats"]["n_blocks"] >= 1


def test_literal_mode_flags_renamed_code_as_diff():
    result = payload(*PAIR_RENAME, mode="literal")
    # so khớp nguyên văn: tên hàm/biến khác nhau → coverage thấp, nhiều token bôi đỏ
    assert result["stats"]["coverage_a"] < 0.5


def test_unrelated_code_has_no_blocks():
    result = payload(*PAIR_UNRELATED, mode="abstract")
    assert result["stats"]["n_blocks"] == 0
    assert result["stats"]["coverage_a"] == 0.0
    assert result["blocks"] == []


def test_min_block_tokens_removes_junk_blocks():
    a = "int f() { ; ; ; }"
    b = "int g() { ; ; ; }"
    small = payload(a, b, mode="literal", min_block_tokens=1)
    big = payload(a, b, mode="literal", min_block_tokens=4)
    assert small["stats"]["n_blocks"] >= 1
    assert big["stats"]["n_blocks"] == 0


def test_payload_shape_is_exact_for_frontend():
    result = payload("int x;", "int y;", mode="literal")
    line = result["a"]["lines"][0]
    kinds = {t["k"] for t in line["tokens"]}
    assert "type" in kinds and "id" in kinds
    # token nào khớp thì b = block id, còn không thì -1 (để frontend bôi đỏ)
    assert all(isinstance(t["b"], int) for t in line["tokens"])
    assert all(t["b"] >= -1 for t in line["tokens"])
    for block in result["blocks"]:
        assert block["a"][0] < block["a"][1]
        assert block["b"][0] < block["b"][1]
        assert block["size"] >= 1


def test_block_ranges_respect_line_bounds():
    a = "int a; int b; int c;"
    b = "int v; int w; int z;"
    result = payload(a, b, mode="abstract")
    assert result["stats"]["n_blocks"] >= 1
    for block in result["blocks"]:
        # [lo, hi) trong giới hạn số dòng hiện có
        assert 0 <= block["a"][0] < block["a"][1] <= len(lex_code(a))
        assert 0 <= block["b"][0] < block["b"][1] <= len(lex_code(b))


def test_empty_side_does_not_crash():
    result = payload("", "int x;", mode="abstract")
    assert result["stats"]["n_blocks"] == 0


def test_blocks_never_overlap_on_each_side():
    a = "int a = 1; int b = 2;"
    b = "int u = 1; int v = 2;"
    p = payload(a, b, mode="abstract")
    for side in ("a", "b"):
        ranges = [tuple(block[side]) for block in p["blocks"]]
        sorted_ranges = sorted(ranges)
        for prev, cur in zip(sorted_ranges, sorted_ranges[1:]):
            assert prev[1] <= cur[0], f"{side} khối chồng nhau: {prev} vs {cur}"