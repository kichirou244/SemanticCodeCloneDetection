"""DatasetCleaner — loại record lỗi có lý do, không drop thầm lặng (Step 5)."""

from dataclasses import dataclass, field

from src.data.loaders.base import CodeSample


@dataclass
class CleaningConfig:
    min_code_chars: int = 10
    expected_language: str = ""  # "" = không kiểm tra ngôn ngữ
    status_allowlist: list[str] | None = None  # None = giữ mọi status + báo cáo
    status_field: str = "status"


@dataclass
class CleaningResult:
    kept: list[CodeSample] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)  # {sample_id, reason}

    def summary(self) -> dict:
        from collections import Counter

        return {
            "n_kept": len(self.kept),
            "n_removed": len(self.removed),
            "reasons": dict(Counter(r["reason"] for r in self.removed)),
        }


class DatasetCleaner:
    """Mọi record bị loại đều ghi reason. Không tự ý lọc accepted-only."""

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    def clean(self, samples: list[CodeSample]) -> CleaningResult:
        out = CleaningResult()
        seen_ids: set[str] = set()
        for s in samples:
            reason = self._check(s, seen_ids)
            if reason is None:
                seen_ids.add(s.sample_id)
                out.kept.append(s)
            else:
                out.removed.append({"sample_id": s.sample_id, "reason": reason})
        return out

    def _check(self, s: CodeSample, seen_ids: set[str]) -> str | None:
        if not (s.code or "").strip():
            return "empty_code"
        if len(s.code.strip()) < self.config.min_code_chars:
            return "too_short"
        if not s.problem_id or s.problem_id == "?":
            return "missing_problem"
        if s.sample_id in seen_ids:
            return "duplicate_submission_id"
        if self.config.expected_language and s.language != self.config.expected_language:
            return "language_mismatch"
        if self.config.status_allowlist is not None:
            status = (s.metadata or {}).get(self.config.status_field, "?")
            if status not in self.config.status_allowlist:
                return f"status_excluded:{status}"
        return None
