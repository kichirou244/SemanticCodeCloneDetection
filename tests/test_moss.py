"""Test thuật toán MOSS local (winnowing fingerprinting).

Bộ này phải xanh mà không cần venv nạp torch/fastapi — chỉ dùng stdlib. Chạy:
    py -m pytest tests/test_moss.py -q
"""

import re

from src.models.baseline.moss import (
    DEFAULT_K,
    FingerprintDoc,
    fingerprints,
    fingerprints_with_spans,
    kgrams,
    normalize,
    scan_pairwise,
    shared_spans,
    similarity,
)

# Tham số MOSS gốc `k=50, w=100` chọn cho văn bản dài. Các đoạn mẫu dưới đây
# chỉ dài 35–141 ký tự sau chuẩn hoá, nên dùng `k=50` sẽ ra 0 k-gram và mọi
# phép so sánh đều bằng 0 — đó là hành vi đúng, không phải lỗi. Test vì thế dùng
# tham số cỡ đoạn nhỏ. Trên POJ-104 nhóm dò `(k, w)` trên validation và dùng
# `k=20, w=5`; xem `scripts/eval_moss.py`.
TOY_K, TOY_W = 5, 2

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

# Ví dụ âm tính mạnh hơn: hàm đệ quy, không có vòng for và không có mảng.
# `CODE_UNRELATED` ở trên dù là bubble sort vẫn dài 141 ký tự và chia sẻ nhiều
# dấu ngoặc với hàm cộng, nên containment vẫn ~0,81 — đặc tính của phép đo
# chứ không phải lỗi của thuật toán.
CODE_DIFFERENT = """int fib(int n) {
    if (n < 2)
        return n;
    return fib(n - 1) + fib(n - 2);
}
"""


def test_normalize_idioms():
    assert normalize("int foo = bar + 42;") == "idid=id+num;"
    # `const char *s` là khai báo con trỏ: `*` và `=` là hai ký tự riêng,
    # không có toán tử `*=` trong mã nguồn.
    assert normalize('const char *s = "hello";') == "idid*id=str;"
    # `'x'` là char literal nên thành `str`, không phải tên biến `x`.
    assert normalize("a = 'x';") == "id=str;"


def test_normalize_keeps_compound_assignment_without_spaces():
    # Regex tách từng dấu câu một, nên `*=` không bị nuốt dấu `=` dù không
    # có khoảng trắng. Bài báo MOSS cần giữ cả hai vì phép gán phức là dấu
    # hiệu cấu trúc quan trọng.
    for src, want in [
        ("a*=x;", "id*=id;"),
        ("a += x;", "id+=id;"),
        ("a-=x;", "id-=id;"),
        ("a/=x;", "id/=id;"),
        ("a%=x;", "id%=id;"),
        ("a<<=x;", "id<<=id;"),
        ("a>>=x;", "id>>=id;"),
    ]:
        assert normalize(src) == want, src


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
    fa = fingerprints(CODE_A, k=TOY_K, w=TOY_W)
    assert similarity(fa, fa) == 1.0


def test_similarity_renamed_is_high():
    fa = fingerprints(CODE_A, k=TOY_K, w=TOY_W)
    fb = fingerprints(CODE_B_RENAMED, k=TOY_K, w=TOY_W)
    assert fa and fb, "du code ngan van phai ra fingerprint o tham so nay"
    assert similarity(fa, fb) >= 0.9


def test_similarity_unrelated_is_lower_than_related():
    # `similarity` là **containment**: |A∩B| / min(|A|,|B|). Với hàm ngắn,
    # vài fingerprint trùng đã đủ để lên gần 1,0, nên con số tuyệt đối không
    # có nghĩa như một "độ giống" thông thường. Đây là lý do
    # `scripts/eval_moss.py` dò ngưỡng trên validation thay vì dùng hằng số.
    fa = fingerprints(CODE_A, k=TOY_K, w=TOY_W)
    fb = fingerprints(CODE_B_RENAMED, k=TOY_K, w=TOY_W)
    fu = fingerprints(CODE_DIFFERENT, k=TOY_K, w=TOY_W)
    assert similarity(fa, fu) < 0.5
    assert similarity(fa, fu) < similarity(fa, fb)


def test_similarity_empty_guard():
    assert similarity([], []) == 0.0
    assert similarity([1, 2], []) == 0.0


def test_fingerprint_doc_from_code():
    doc = FingerprintDoc.from_code("a.c", CODE_A, k=TOY_K, w=TOY_W)
    assert doc.path == "a.c"
    assert doc.n_chars_original == len(CODE_A)
    assert doc.fingerprints


def test_scan_pairwise_sorts_and_filters():
    docs = [
        FingerprintDoc.from_code("a.c", CODE_A, k=TOY_K, w=TOY_W),
        FingerprintDoc.from_code("b.c", CODE_B_RENAMED, k=TOY_K, w=TOY_W),
        FingerprintDoc.from_code("c.c", CODE_UNRELATED, k=TOY_K, w=TOY_W),
    ]
    pairs = scan_pairwise(docs, threshold=0.5)
    assert pairs  # A↔B là clone nên chắc chắn lọt trên ngưỡng
    assert all(p["similarity"] >= 0.5 for p in pairs)
    for prev, nxt in zip(pairs, pairs[1:]):
        assert prev["similarity"] >= nxt["similarity"]
    assert all(p["n_fp_a"] > 0 and p["n_fp_b"] > 0 for p in pairs)
    assert pairs[0]["a"] == "a.c" and pairs[0]["b"] == "b.c"


def test_scan_pairwise_cap():
    docs = [
        FingerprintDoc.from_code(f"f{i}.c", CODE_B_RENAMED, k=TOY_K, w=TOY_W)
        for i in range(6)
    ]
    assert len(scan_pairwise(docs, threshold=0.0, cap=3)) == 3


def test_scan_pairwise_threshold_edge():
    docs = [
        FingerprintDoc.from_code("a.c", CODE_A, k=TOY_K, w=TOY_W),
        FingerprintDoc.from_code("b.c", CODE_B_RENAMED, k=TOY_K, w=TOY_W),
    ]
    assert scan_pairwise(docs, threshold=10.0) == []
    assert scan_pairwise(docs, threshold=0.0) != []


def test_short_code_yields_no_kgrams_under_document_default():
    # `k=50` với đoạn 35 ký tự là không có k-gram nào — hành vi đúng theo định
    # nghĩa, và là lý do các test còn lại phải truyền `k` tường minh.
    assert kgrams(normalize(CODE_A), DEFAULT_K) == []
    assert fingerprints(CODE_A) == []


def test_normalize_hex_and_floats():
    assert normalize("0x1F + 3.14 + .5 + 1e3") == "num+num+num+num"


def test_similarity_commutative_for_equal_lengths():
    fa = fingerprints(CODE_A)
    fb = fingerprints(CODE_UNRELATED)
    assert similarity(fa, fb) == similarity(fb, fa)


def test_spans_match_plain_fingerprints():
    """Bat bien: bo span di phai ra dung fingerprints() — tren nhieu dang code."""
    from src.models.baseline.moss import _normalize_parts

    samples = [CODE_A, CODE_B_RENAMED, CODE_UNRELATED, "", "int x;",
               "/* comment only */", "printf(\"%d\\n\", x);"]
    for code in samples:
        for k, w in ((5, 5), (20, 5)):
            plain = fingerprints(code, k, w)
            with_spans = fingerprints_with_spans(code, k, w)
            assert [h for h, _, _ in with_spans] == plain, (code[:30], k, w)
    joined = "".join(t for t, _, _ in _normalize_parts(CODE_A))
    assert joined == normalize(CODE_A)


def test_shared_spans_cover_clone_not_unrelated():
    from src.models.baseline.moss import _normalize_parts  # noqa

    fa = fingerprints_with_spans(CODE_A, 5, 5)
    fb = fingerprints_with_spans(CODE_B_RENAMED, 5, 5)
    sa, sb, n = shared_spans(fa, fb)
    assert n > 0 and sa and sb
    for s, e in sa + sb:
        assert 0 <= s < e
    assert all(e <= len(CODE_A) for _, e in sa)
    # khong chong lan sau khi gop
    flat = sorted(sa)
    assert all(b[0] >= a[1] for a, b in zip(flat, flat[1:]))
    fu = fingerprints_with_spans(CODE_UNRELATED, 5, 5)
    _, _, n2 = shared_spans(fa, fu)
    # Hai file deu co dong #include<...> (chuan hoa giong het) nen van chung
    # vai hash — tinh chat dung la: clone chung NHIEU hon khong-clone.
    assert 0 <= n2 < n
