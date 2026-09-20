"""Kiểm tra schema + thống kê audit dataset (số mẫu, phân bố problem, missing)."""

import statistics
from collections import Counter

from src.data.loaders.base import CodeSample


def audit_samples(samples: list[CodeSample]) -> dict:
    problems = Counter(s.problem_id for s in samples)
    missing_code = sum(1 for s in samples if not (s.code or "").strip())
    sizes = sorted(problems.values())
    return {
        "n_samples": len(samples),
        "n_problems": len(problems),
        "missing_code": missing_code,
        "top_problems": problems.most_common(10),
        "problem_size_stats": {
            "min": sizes[0] if sizes else 0,
            "max": sizes[-1] if sizes else 0,
            "mean": round(statistics.mean(sizes), 2) if sizes else 0,
            "median": statistics.median(sizes) if sizes else 0,
            "n_singleton": sum(1 for v in sizes if v == 1),
            "n_lt5": sum(1 for v in sizes if v < 5),
        },
        "problem_counts": dict(problems),
    }
