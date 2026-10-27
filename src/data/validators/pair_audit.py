"""Audit và kiểm tra tính toàn vẹn, phân phối, rò rỉ (leakage) cho dataset cặp mã nguồn."""

from __future__ import annotations

from collections import Counter
import hashlib
import statistics
from typing import Any, Iterable

from src.data.loaders.base import ClonePairSample


def compute_code_hash(code: str) -> str:
    """Tính hash MD5 sau khi strip khoảng trắng cơ bản để dò code trùng lặp."""
    cleaned = "".join(code.split())
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def audit_pairs(pairs: list[ClonePairSample]) -> dict[str, Any]:
    """Kiểm tra schema, phân phối nhãn, độ dài và các bất thường trong tập cặp mã nguồn."""
    if not pairs:
        return {
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
            "positive_ratio": 0.0,
            "unique_func_ids": 0,
            "missing_code": 0,
            "stats": {},
        }

    n_positive = sum(1 for p in pairs if p.label == 1)
    n_negative = sum(1 for p in pairs if p.label == 0)
    pos_ratio = round(n_positive / len(pairs), 4)

    unique_func_ids = set()
    missing_code = 0

    f1_lines: list[int] = []
    f2_lines: list[int] = []
    f1_chars: list[int] = []
    f2_chars: list[int] = []

    canonical_pairs: dict[str, set[int]] = {}
    duplicate_count = 0

    for p in pairs:
        if not (p.func1 and p.func1.strip()):
            missing_code += 1
        if not (p.func2 and p.func2.strip()):
            missing_code += 1

        if p.id1:
            unique_func_ids.add(p.id1)
        if p.id2:
            unique_func_ids.add(p.id2)

        # Đo độ dài dòng & ký tự
        c1 = p.func1 or ""
        c2 = p.func2 or ""
        f1_lines.append(c1.count("\n") + 1 if c1 else 0)
        f2_lines.append(c2.count("\n") + 1 if c2 else 0)
        f1_chars.append(len(c1))
        f2_chars.append(len(c2))

        # Kiểm tra trùng lặp cặp canonical (không phân biệt thứ tự id1, id2)
        lo, hi = (p.id1, p.id2) if p.id1 <= p.id2 else (p.id2, p.id1)
        canon_key = f"{lo}:::{hi}"
        if canon_key in canonical_pairs:
            duplicate_count += 1
            canonical_pairs[canon_key].add(p.label)
        else:
            canonical_pairs[canon_key] = {p.label}

    conflicts = sum(1 for labels in canonical_pairs.values() if len(labels) > 1)

    all_lines = f1_lines + f2_lines
    all_chars = f1_chars + f2_chars

    return {
        "n_pairs": len(pairs),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_ratio": pos_ratio,
        "unique_func_ids": len(unique_func_ids),
        "missing_code": missing_code,
        "duplicate_pairs": duplicate_count,
        "conflicting_pairs": conflicts,
        "length_stats": {
            "lines_min": min(all_lines) if all_lines else 0,
            "lines_max": max(all_lines) if all_lines else 0,
            "lines_mean": round(statistics.mean(all_lines), 2) if all_lines else 0,
            "lines_median": statistics.median(all_lines) if all_lines else 0,
            "chars_min": min(all_chars) if all_chars else 0,
            "chars_max": max(all_chars) if all_chars else 0,
            "chars_mean": round(statistics.mean(all_chars), 2) if all_chars else 0,
            "chars_median": statistics.median(all_chars) if all_chars else 0,
        },
    }


def check_pair_leakage(splits: dict[str, list[ClonePairSample]]) -> dict[str, Any]:
    """Kiểm tra nguy cơ rò rỉ dữ liệu (leakage) giữa các split (train, validation, test).

    Bao gồm:
      1. Pair leakage: Cặp (id1, id2) giống nhau xuất hiện ở 2 split khác nhau.
      2. Function leakage: Một hàm (id1 hoặc id2) xuất hiện đồng thời ở train và val/test.
      3. Code hash leakage: Code giống hệt nhau (sau khi xóa whitespace) xuất hiện xuyên split.
    """
    split_names = list(splits.keys())
    pair_maps: dict[str, set[str]] = {}
    func_maps: dict[str, set[str]] = {}
    hash_maps: dict[str, set[str]] = {}

    for name, sample_list in splits.items():
        pairs_set: set[str] = set()
        funcs_set: set[str] = set()
        hashes_set: set[str] = set()
        for p in sample_list:
            lo, hi = (p.id1, p.id2) if p.id1 <= p.id2 else (p.id2, p.id1)
            pairs_set.add(f"{lo}:::{hi}")
            if p.id1:
                funcs_set.add(p.id1)
            if p.id2:
                funcs_set.add(p.id2)
            if p.func1:
                hashes_set.add(compute_code_hash(p.func1))
            if p.func2:
                hashes_set.add(compute_code_hash(p.func2))

        pair_maps[name] = pairs_set
        func_maps[name] = funcs_set
        hash_maps[name] = hashes_set

    pair_overlaps: dict[str, int] = {}
    func_overlaps: dict[str, int] = {}
    hash_overlaps: dict[str, int] = {}

    has_leakage = False

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            key = f"{s1}_vs_{s2}"

            p_overlap = len(pair_maps[s1] & pair_maps[s2])
            f_overlap = len(func_maps[s1] & func_maps[s2])
            h_overlap = len(hash_maps[s1] & hash_maps[s2])

            pair_overlaps[key] = p_overlap
            func_overlaps[key] = f_overlap
            hash_overlaps[key] = h_overlap

            # Pair overlap xuyên split là vi phạm nghiêm trọng (train test contamination)
            if p_overlap > 0:
                has_leakage = True

    return {
        "splits": {name: len(samples) for name, samples in splits.items()},
        "pair_overlaps": pair_overlaps,
        "function_overlaps": func_overlaps,
        "code_hash_overlaps": hash_overlaps,
        "has_critical_pair_leakage": has_leakage,
        "verdict": "FAIL - Pair Leakage Detected!" if has_leakage else "PASS - No Pair Leakage",
    }
