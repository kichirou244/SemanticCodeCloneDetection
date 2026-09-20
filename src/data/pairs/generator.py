"""PairGenerator — sinh clone/non-clone pairs deterministic (Steps 12-17).

- Positive: cùng problem, cap số pair/problem (tránh problem lớn dominate).
- Negative: cặp problems khác nhau, tỉ lệ 1:1 mặc định, random/balanced.
- Pair ID chuẩn tắc vô thứ tự: (min_id, max_id) — A,B ≡ B,A (Step 19).
- Không oversample test: mỗi split sinh độc lập với cap riêng.
"""

import itertools
import random
from collections import defaultdict
from dataclasses import dataclass, field

from src.data.loaders.base import CodeSample


@dataclass
class PairRecord:
    pair_id: str
    code_a_id: str
    code_b_id: str
    problem_a: str
    problem_b: str
    label: int
    split: str


@dataclass
class PairGenConfig:
    seed: int = 42
    negative_ratio: int = 1  # số negative mỗi positive
    max_positive_pairs_per_problem: int = 2000
    max_positive_pairs_per_problem_val_test: int = 500


class PairGenerator:
    def __init__(self, config: PairGenConfig | None = None):
        self.config = config or PairGenConfig()

    @staticmethod
    def canonical_pair_id(a: str, b: str) -> str:
        lo, hi = (a, b) if a <= b else (b, a)
        return f"{lo}:::{hi}"

    def generate(self, samples: list[CodeSample], split: str) -> list[PairRecord]:
        cfg = self.config
        rng = random.Random(f"{cfg.seed}/{split}")
        by_problem: dict[str, list[CodeSample]] = defaultdict(list)
        for s in samples:
            by_problem[s.problem_id].append(s)

        cap = (
            cfg.max_positive_pairs_per_problem
            if split == "train"
            else cfg.max_positive_pairs_per_problem_val_test
        )
        positives: list[PairRecord] = []
        for pid in sorted(by_problem):
            ids = sorted(s.sample_id for s in by_problem[pid])
            all_pairs = list(itertools.combinations(ids, 2))
            if len(all_pairs) > cap:
                all_pairs = rng.sample(all_pairs, cap)
            for i, (a, b) in enumerate(all_pairs):
                positives.append(
                    PairRecord(
                        pair_id=f"{split}-pos-{pid}-{i}",
                        code_a_id=a,
                        code_b_id=b,
                        problem_a=pid,
                        problem_b=pid,
                        label=1,
                        split=split,
                    )
                )

        # Negative: cặp problems khác nhau, số lượng = ratio × positives.
        problems = sorted(by_problem)
        prob_of = {s.sample_id: s.problem_id for s in samples}
        used: set[str] = {self.canonical_pair_id(p.code_a_id, p.code_b_id) for p in positives}
        negatives: list[PairRecord] = []
        need = len(positives) * cfg.negative_ratio
        guard = 0
        i = 0
        while len(negatives) < need and guard < need * 50 + 1000:
            guard += 1
            pa, pb = rng.sample(problems, 2)
            a = rng.choice(by_problem[pa]).sample_id
            b = rng.choice(by_problem[pb]).sample_id
            key = self.canonical_pair_id(a, b)
            if key in used:
                continue
            used.add(key)
            negatives.append(
                PairRecord(
                    pair_id=f"{split}-neg-{i}",
                    code_a_id=a,
                    code_b_id=b,
                    problem_a=prob_of[a],
                    problem_b=prob_of[b],
                    label=0,
                    split=split,
                )
            )
            i += 1
        return positives + negatives
