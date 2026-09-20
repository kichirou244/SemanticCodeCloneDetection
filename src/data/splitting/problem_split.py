"""Split theo problem/task identity — chống leakage cùng problem sang test."""

import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.loaders.base import CodeSample


def filter_eligible(
    samples: list[CodeSample], min_samples_per_problem: int = 2
) -> tuple[list[CodeSample], list[dict]]:
    """Tách problems đủ điều kiện tạo pair. Không drop thầm lặng (Step 11)."""
    by_problem: dict[str, list[CodeSample]] = defaultdict(list)
    for s in samples:
        by_problem[s.problem_id].append(s)
    kept, ineligible = [], []
    for pid in sorted(by_problem):
        if len(by_problem[pid]) >= min_samples_per_problem:
            kept.extend(by_problem[pid])
        else:
            ineligible.append(
                {
                    "problem_id": pid,
                    "n_samples": len(by_problem[pid]),
                    "reason": f"too_few_samples (<{min_samples_per_problem})",
                }
            )
    return kept, ineligible


def split_by_problem(
    samples: list[CodeSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[CodeSample]]:
    """Các problem trong test không bao giờ xuất hiện ở train/val."""
    by_problem: dict[str, list[CodeSample]] = defaultdict(list)
    for s in samples:
        by_problem[s.problem_id].append(s)
    problems = sorted(by_problem)
    rng = random.Random(seed)
    rng.shuffle(problems)
    n = len(problems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    groups = {
        "train": problems[:n_train],
        "validation": problems[n_train : n_train + n_val],
        "test": problems[n_train + n_val :],
    }
    out: dict[str, list[CodeSample]] = {}
    for split, plist in groups.items():
        subset = [s for p in plist for s in by_problem[p]]
        for s in subset:
            s.split = split
        out[split] = subset
    return out


def save_problem_ids(splits: dict[str, list[CodeSample]], out_dir: str | Path) -> dict:
    """Lưu problem IDs + sample IDs từng split để tái tạo (Step 10)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, samples in splits.items():
        payload = {
            "problems": sorted({s.problem_id for s in samples}),
            "sample_ids": sorted(s.sample_id for s in samples),
        }
        path = out_dir / f"{split}_problems.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths[split] = str(path)
    return paths
