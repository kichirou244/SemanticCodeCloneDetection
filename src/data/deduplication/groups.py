"""DuplicateDetector — nhóm duplicate, chọn canonical deterministic (Steps 7-8).

- Exact group: hash(raw code).
- Normalized group: hash(normalized code), chỉ nhóm các mẫu chưa trùng exact.
- Canonical = sample_id nhỏ nhất trong nhóm (deterministic, không random).
- Dedup PHẢI chạy trước problem split (DEC-010) để chặn cross-problem leakage.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from src.data.deduplication.duplicates import fingerprint
from src.data.loaders.base import CodeSample


@dataclass
class DedupResult:
    canonical: list[CodeSample] = field(default_factory=list)
    groups: list[dict] = field(default_factory=list)
    dup_to_canonical: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "n_input": len(self.canonical) + len(self.dup_to_canonical),
            "n_canonical": len(self.canonical),
            "n_removed": len(self.dup_to_canonical),
            "n_groups_exact": sum(1 for g in self.groups if g["kind"] == "exact"),
            "n_groups_normalized": sum(
                1 for g in self.groups if g["kind"] == "normalized"
            ),
        }


class DuplicateDetector:
    def detect(
        self, samples: list[CodeSample], normalized: bool = True
    ) -> DedupResult:
        result = DedupResult()
        removed: set[str] = set()

        for kind in ("exact", "normalized"):
            buckets: dict[str, list[CodeSample]] = defaultdict(list)
            for s in samples:
                if s.sample_id in removed:
                    continue
                text = s.normalized_code if (normalized and kind == "normalized") else s.code
                buckets[fingerprint(text, normalized=False)].append(s)
            for members in buckets.values():
                if len(members) < 2:
                    continue
                members = sorted(members, key=lambda s: s.sample_id)
                canon, dups = members[0], members[1:]
                result.groups.append(
                    {
                        "kind": kind,
                        "canonical_id": canon.sample_id,
                        "member_ids": [m.sample_id for m in members],
                    }
                )
                for d in dups:
                    removed.add(d.sample_id)
                    result.dup_to_canonical[d.sample_id] = canon.sample_id

        kept_ids = {s.sample_id for s in samples} - removed
        result.canonical = [s for s in samples if s.sample_id in kept_ids]
        return result
