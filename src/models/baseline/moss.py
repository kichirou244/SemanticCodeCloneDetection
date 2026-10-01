"""Winnowing fingerprinting — thuật toán của MOSS (Sleator & Bernstein 2003).

Lý do dùng làm baseline: MOSS là công cụ phát hiện đạo văn chuẩn của Stanford,
nhưng service thật (`moss.stanford.edu`) cần tài khoản và gửi code lên mạng.
Nghiên cứu cần baseline **lặp lại được**, nên ở đây chúng tôi triển khai đúng
thuật toán fingerprinting của nó, chạy local, không phụ thuộc mạng. Đây là
cách mà chính bài báo MOSS mô tả, và cũng là cách các bản cài lại (pymoss,
plaggie...) triển khai.

Ba bước của thuật toán:

1. **Chuẩn hoá** — biến mọi thứ thành một chuỗi không có khoảng trắng:
   - identifier (`foo`, `MAX_LEN`, ...) → `id`  ⇒ clone bị đổi tên biến vẫn trùng.
   - số (`42`, `1e-3`, ...) → `num`; chuỗi/char literal → `str`.
   - comment (`//` và `/* ... */`) bị lược bỏ.
   - symbol giữ nguyên ⇒ dấu cú pháp vẫn neo được vị trí khớp.
2. **k-gram + window**: cắt chuỗi trên thành các k-gram liên tiếp, với mỗi cửa
   sổ `w` k-gram chọn ra gram có giá trị hash nhỏ nhất (fingerprint). MOSS dùng
   k=50, w=100 — đúng giá trị mặc định ở đây.
3. **So trùng**: hai tài liệu giống nhau bao nhiêu = fingerprint trùng nhau bao
   nhiêu (containment: |FP∩FP| / min).

Về tính ổn định của hash: không dùng `hash()` Python (hạt muối ngẫu nhiên mỗi
lần chạy ⇒ fingerprint thay đổi theo tiến trình). Dùng `zlib.crc32` — tất định
theo nội dung, đủ chống đụng độ cho demo, nhanh và không cần rolling hash.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass

#: MOSS dùng k=50, w=100. Các giá trị này được dùng như mặc định ở mọi nơi,
#: kể cả demo scan, để con số % trên UI khớp với F1 trong báo cáo.
DEFAULT_K = 50
DEFAULT_W = 100

#: Nhận diện các lớp token. `re.DOTALL` để comment khối span nhiều dòng.
_TOKEN_RE = re.compile(
    r"""
      (?P<comment>//[^\n]*|/\*.*?\*/)
    | (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
    | (?P<num>\.?[0-9][A-Za-z0-9_.]*(?:[eEpP][+-]?[0-9]+)?)
    | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<punct>[^\s])
    """,
    re.VERBOSE | re.DOTALL,
)


def normalize(code: str) -> str:
    """Chuẩn hoá code thành một chuỗi không khoảng trắng theo quy tắc MOSS.

    Ví dụ `int foo = bar + 42;` → `id id = id + num;`. Comment biến mất hoàn
    toàn; chuỗi literal và số chỉ còn là "str"/"num" nên copy-and-paste đổi
    hằng số vẫn bắt được.
    """
    out: list[str] = []
    for match in _TOKEN_RE.finditer(code):
        kind = match.lastgroup
        if kind == "comment":
            continue
        if kind == "word":
            out.append("id")
        elif kind == "num":
            out.append("num")
        elif kind == "str":
            out.append("str")
        else:  # punct: giữ nguyên symbol
            out.append(match.group())
    return "".join(out)


def kgrams(stream: str, k: int = DEFAULT_K) -> list[str]:
    """Các k-gram liên tiếp của chuỗi đã chuẩn hoá. Gram cuối thiếu thì bỏ."""
    return [stream[i : i + k] for i in range(len(stream) - k + 1)] if k > 0 else []


def _stable_hash(gram: str) -> int:
    # crc32 tất định; & 0xFFFFFFFF để luôn dương.
    return zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF


def fingerprints(
    code: str,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
) -> list[int]:
    """Danh sách fingerprint (giá trị hash) của một tài liệu.

    Mỗi cửa sổ `w` k-gram liên tiếp đóng góp giá trị hash nhỏ nhất của nó — đó
    là "điểm mốc" mà MOSS dùng. Gộp các giá trị liền nhau trùng lặp (thường xảy
    ra khi vài cửa sổ liên tiếp cùng chọn một gram) để fingerprint không bị đọc
    quá nặng, giống MOSS.
    """
    if not code or not code.strip():
        return []
    grams = kgrams(normalize(code), k)
    if not grams:
        return []
    win = min(w, max(1, len(grams)))
    values: list[int] = []
    for i in range(len(grams) - win + 1):
        values.append(min(_stable_hash(g) for g in grams[i : i + win]))
    # gộp các giá trị trùng lặp liền nhau
    out: list[int] = []
    for v in values:
        if not out or out[-1] != v:
            out.append(v)
    return out


def similarity(
    fingerprint_a: list[int],
    fingerprint_b: list[int],
) -> float:
    """Containment: số fingerprint chung chia cho số fingerprint của bên ít hơn.

    Chuẩn hoá chia cho `min(n_a, n_b)` nên kết quả đối xứng `0..1` và không phạt
    tài liệu dài hơn (đoạn chung dễ nằm ráp trong một file nhỏ). Giống JPlag
    family: báo "phần TÀI LIỆU bị lấy ra" thay vì trung bình hai chiều.
    """
    if not fingerprint_a or not fingerprint_b:
        return 0.0
    count_a = Counter(fingerprint_a)
    count_b = Counter(fingerprint_b)
    matches = sum(
        min(ca, count_b.get(value, 0)) for value, ca in count_a.items()
    )
    total = min(sum(count_a.values()), sum(count_b.values()))
    return matches / total if total else 0.0


@dataclass(frozen=True)
class FingerprintDoc:
    """Tài liệu đã fingerprint hoá, sẵn để so cặp trong scan thư mục."""

    path: str
    fingerprints: list[int]
    n_chars_original: int
    n_kgrams: int

    @classmethod
    def from_code(cls, path: str, code: str, k: int = DEFAULT_K, w: int = DEFAULT_W) -> "FingerprintDoc":
        return cls(path=path, fingerprints=fingerprints(code, k, w),
                   n_chars_original=len(code),
                   n_kgrams=len(kgrams(normalize(code), k)))


def scan_pairwise(docs: list[FingerprintDoc], threshold: float, cap: int = 500) -> list[dict]:
    """So mọi cặp trong tập tài liệu, trả về các cặp >= `threshold`, giảm dần.

    Kết quả mỗi cặp: `{a, b, similarity, n_fp_a, n_fp_b, n_matches}`. `cap` giới
    hạn số cặp trả về để response không phình khi database lớn.
    """
    out: list[dict] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            sim = similarity(docs[i].fingerprints, docs[j].fingerprints)
            if sim >= threshold:
                out.append({
                    "a": docs[i].path,
                    "b": docs[j].path,
                    "similarity": round(sim, 4),
                    "n_fp_a": len(docs[i].fingerprints),
                    "n_fp_b": len(docs[j].fingerprints),
                })
    out.sort(key=lambda pair: (-pair["similarity"], pair["a"], pair["b"]))
    return out[:cap]