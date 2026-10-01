"""Test thuật toán MOSS local (winnowing fingerprinting).

Bộ này phải xanh mà không cần venv nạp torch/fastapi — chỉ dùng stdlib. Chạy:
    py -m pytest tests/test_moss.py -q
"""

import re

from src.models.baseline.moss import (
    FingerprintDoc,
    fingerprints,
    kgrams,
    normalize,
    scan_pairwise,
    similarity,
)

CODE_A = """#include <stdio.h>
int add(int a, int b) {
    // cộng hai số
    return a + b;
}
"""

# Chỉ đổi tên biến + comment — copy hầu như nguyên vẹn.
CODE_B_RENAMED = """#include <stdio.h>
int add(int x, int y) {
    /* hàm này cộng */
    return x + y;
}
"""

CODE_UNRELATED = """#include <string.h>
void sort_array(int *arr, int n) {
    int i, j, tmp;
    for (i = 0; i < n - 1; i++)
        for (j = i + 1; j < n; j++)
            if (arr[j] < arr[i]) { tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
}
"""


def test_normalize_idioms():
    assert normalize("int foo = bar + 42;") == "idid=id+num;"
    assert normalize('const char *s = "hello";') == "idid*=id=str;"
    assert normalize("a = 'x';") == "id=id=str;"


def test_normalize_strips_block_comment_across_lines():
    src = "int f() { /* multi\nline */ return 1; }"
    assert "/" not in normalize(src)
    assert normalize(src) == "idid(){idnum;}"
    assert "multi" not in normalize(src)
    assert "line" not in normalize(src)


def test_normalize_strips_line_comment_midline():
    assert normalize("int a = 1; // tạm\nint b = a + 1;") == "idid=num;idid=id+num;"


def test_normalize_treats_dataless_literals_as_markers():
    # Số và chuỗi bị dinghy hoá — copy paste đổi hằng số vẫn bắt được.
    assert "42" not in normalize('return "hello" * 42')
    assert normalize('return "hello" * 42') == "idstr*num"


def test_kgrams_basic():
    assert kgrams("abcde", k=3) == ["abc", "bcd", "cde"]
    assert kgrams("abcde", k=3)[0] == "abc"
    assert kgrams("ab", k=3) == []
    assert kgrams("", k=3) == []


def test_fingerprints_deterministic_across_calls():
    assert fingerprints(CODE_A) == fingerprints(CODE_A)


def test_fingerprints_deterministic_across_processes():
    import subprocess
    import sys
    from pathlib import Path

    here = Path(__file__).resolve()
    code_hash = str(fingerprints(CODE_A))
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from src.models.baseline.moss import fingerprints; "
        f"print(fingerprints({CODE_A!r}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == code_hash


def test_fingerprints_empty():
    assert fingerprints("") == []
    assert fingerprints("   \n  ") == []
    assert fingerprints("... solitary") == []


def test_similarity_identical_is_1():
    assert similarity(fingerprints(CODE_A), fingerprints(CODE_A)) == 1.0


def test_similarity_renamed_is_high():
    sim = similarity(fingerprints(CODE_A), fingerprints(CODE_B_RENAMED))
    assert sim >= 0.9


def test_similarity_unrelated_is_low():
    sim = similarity(fingerprints(CODE_A), fingerprints(CODE_UNRELATED))
    assert sim < 0.2


def test_similarity_empty_guard():
    assert similarity([], []) == 0.0
    assert similarity([1, 2], []) == 0.0


def test_fingerprint_doc_from_code():
    doc = FingerprintDoc.from_code("a.c", CODE_A)
    assert doc.path == "a.c"
    assert doc.n_chars_original == len(CODE_A)
    assert doc.fingerprints


def test_scan_pairwise_sorts_and_filters():
    docs = [
        FingerprintDoc.from_code("a.c", CODE_A),
        FingerprintDoc.from_code("b.c", CODE_B_RENAMED),
        FingerprintDoc.from_code("c.c", CODE_UNRELATED),
    ]
    pairs = scan_pairwise(docs, threshold=0.5)
    assert pairs  # A↔B là clone nên chắc chắn lọt trên ngưỡng
    assert all(p["similarity"] >= 0.5 for p in pairs)
    for prev, nxt in zip(pairs, pairs[1:]):
        assert prev["similarity"] >= nxt["similarity"]
    assert all(p["n_fp_a"] > 0 and p["n_fp_b"] > 0 for p in pairs)
    assert pairs[0]["a"] == "a.c" and pairs[0]["b"] == "b.c"


def test_scan_pairwise_cap():
    docs = [FingerprintDoc.from_code(f"f{i}.c", CODE_B_RENAMED) for i in range(6)]
    assert len(scan_pairwise(docs, threshold=0.0, cap=3)) == 3


def test_scan_pairwise_threshold_edge():
    docs = [
        FingerprintDoc.from_code("a.c", CODE_A),
        FingerprintDoc.from_code("b.c", CODE_B_RENAMED),
    ]
    assert scan_pairwise(docs, threshold=10.0) == []
    assert scan_pairwise(docs, threshold=0.0) != []


def test_normalize_hex_and_floats():
    assert normalize("0x1F + 3.14 + .5 + 1e3") == "num+num+num+num"


def test_similarity_commutative_for_equal_lengths():
    fa = fingerprints(CODE_A)
    fb = fingerprints(CODE_UNRELATED)
    assert similarity(fa, fb) == similarity(fb, fa)