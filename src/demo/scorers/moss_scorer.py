"""B1 — MOSS (winnowing fingerprinting) làm baseline.

Thay TF-IDF: cùng thuật toán fingerprinting của MOSS (Sleator & Bernstein 2003),
triển khai local tại `src/models/baseline/moss.py`, dùng chung cho cả nghiên cứu
(EXP-006) lẫn demo scan — nên con số trên UI đồng nhất với F1 trong báo cáo.

Khác các scorer neural, MOSS không cần nạp model: `_load` là no-op, `score` trả
ngay. Threshold vẫn đọc từ `threshold.json` (EXP-006 chọn trên validation). Chỉ
khi chưa chạy EXP-006 thì verdict về `unknown`, không tự bịa ngưỡng.
"""

from __future__ import annotations

from pathlib import Path

from src.demo.scorers.base import BaseScorer
from src.models.baseline.moss import DEFAULT_K, DEFAULT_W, fingerprints, similarity


class MossScorer(BaseScorer):
    """Một cặp code → containment fingerprint MOSS trong [0, 1]."""

    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        k: int = DEFAULT_K,
        w: int = DEFAULT_W,
        device: str = "auto",
    ):
        super().__init__(model_id, label, results_dir, device)
        self.k = k
        self.w = w

    def checkpoint_hint(self) -> str:
        return f"MOSS winnowing · k={self.k} w={self.w} · không cần checkpoint"

    def _load(self) -> None:
        # Không có gì để nạp — fingerprint tính tất định ngay trong `_infer`.
        return None

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        fa = fingerprints(code_a, k=self.k, w=self.w)
        fb = fingerprints(code_b, k=self.k, w=self.w)
        score = similarity(fa, fb)
        return score, {
            "n_fp_a": len(fa),
            "n_fp_b": len(fb),
            "k": self.k,
            "w": self.w,
        }