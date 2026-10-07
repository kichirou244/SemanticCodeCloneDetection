"""B1 — MOSS làm baseline.

Có hai chế độ, chọn TỰ ĐỘNG khi nạp:

- `moss-service` (mặc định nếu có `MOSS_USER_ID`): gọi **MOSS thật**
  (moss.stanford.edu) qua `src/models/baseline/moss_service.py` — nộp code, lấy
  HTML report, trả kèm `report_url` để UI hiện link kết quả gốc của Stanford.
  Điểm một cặp = `max(pct_a, pct_b)/100`; cặp không được MOSS gắn cờ = 0.
- `moss-local` (fallback khi chưa khai `MOSS_USER_ID`): triển khai local cùng
  thuật toán winnowing tại `src/models/baseline/moss.py` (k=20, w=5 — tham số
  dò trên validation EXP-013). Giữ để demo chạy offline và test deterministic.

Cả hai chế độ dùng chung `results_dir` → đọc ngưỡng từ `threshold.json`
(EXP-014: rule `score > 0.01` = MOSS gắn cờ). Nộp MOSS thật được cache theo hash
nội dung (`cache_dir`) nên chấm lại cùng cặp không tốn thêm lượt nộp Stanford.
"""

from __future__ import annotations

from pathlib import Path

from src.demo.scorers.base import BaseScorer
from src.models.baseline.moss import (
    DEFAULT_K,
    DEFAULT_W,
    fingerprints,
    fingerprints_with_spans,
    shared_spans,
    similarity,
)


class MossScorer(BaseScorer):
    """Một cặp code → điểm MOSS trong [0, 1] (service thật hoặc local fallback)."""

    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        k: int = DEFAULT_K,
        w: int = DEFAULT_W,
        cache_dir: Path | None = None,
        show: int = 1000,
        max_matches: int = 1000,
        device: str = "auto",
    ):
        super().__init__(model_id, label, results_dir, device)
        self.k = k
        self.w = w
        self.cache_dir = cache_dir
        self.show = show
        self.max_matches = max_matches
        self._client = None  # MossServiceClient nếu remote mode

    @property
    def mode(self) -> str:
        return "moss-service" if self._client is not None else "moss-local"

    def checkpoint_hint(self) -> str:
        if self.mode == "moss-service":
            return "MOSS service (moss.stanford.edu) · code nộp thật, có link report"
        return "MOSS winnowing local (chưa có MOSS_USER_ID) · k=%d w=%d" % (self.k, self.w)

    def _load(self) -> None:
        # Remote cần MOSS_USER_ID; thiếu thì FALLBACK local (không fail cứng) —
        # demo chạy được offline, người chưa đăng ký MOSS vẫn dùng được.
        from src.models.baseline.moss_service import MossServiceClient

        try:
            self._client = MossServiceClient(
                cache_dir=self.cache_dir, show=self.show, max_matches=self.max_matches
            )
        except Exception:  # noqa: BLE001 — thiếu id => chạy local, không chết
            self._client = None

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        if self.mode == "moss-service":
            return self._infer_service(code_a, code_b)
        return self._infer_local(code_a, code_b)

    # ----------------------------------------------------------------- service
    def _infer_service(self, code_a: str, code_b: str) -> tuple[float, dict]:
        report = self._client.submit({"a": code_a, "b": code_b})
        score = report.score("a", "b")
        match = report._by_pair.get(frozenset(("a", "b")))
        return score, {
            "mode": "moss-service",
            "report_url": report.url,
            "pct_a": match.pct_a if match else None,
            "pct_b": match.pct_b if match else None,
            "n_matches": len(report.matches),
            "show": self.show,
        }

    def score_all_pairs(
        self, items: list[tuple[str, str]]
    ) -> dict[tuple[str, str], tuple[float | None, dict]]:
        """Chấm all-pairs bằng MỘT lần nộp MOSS (không phải N² lần).

        `items = [(tên_file, code)]`; trả `{ (a, b): (điểm, detail) }` cho mọi
        cặp — cùng khuôn với `CpgScorer.score_all_pairs`. Service mode: nộp cả
        tập file, cặp không nằm trong report (không bị gắn cờ) = 0. Local mode:
        vòng per-pair để quét vẫn chạy offline.
        """
        names = [name for name, _ in items]
        keys = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        out: dict[tuple[str, str], tuple[float | None, dict]] = {}
        if not keys:
            return out
        if self.mode != "moss-service":
            code_by = dict(items)
            return {k: self.score_raw(code_by[k[0]], code_by[k[1]]) for k in keys}
        if not self.ensure_loaded():
            err = self._error or "load failed"
            return {k: (None, {"error": err}) for k in keys}
        code_by = dict(items)
        key_of = {name: f"{i:04d}" for i, name in enumerate(names)}
        files = {key_of[name]: code_by[name] for name in names}
        report = self._client.submit(files)
        base_detail = {"mode": "moss-service", "report_url": report.url,
                       "n_matches": len(report.matches)}
        return {k: (report.score(key_of[k[0]], key_of[k[1]]), dict(base_detail))
                for k in keys}

    def evidence(self, code_a: str, code_b: str) -> dict | None:
        if self.mode == "moss-service":
            rep = self._client.submit({"a": code_a, "b": code_b}, with_details=True)
            ev = rep.evidence("a", "b", code_a, code_b)
            if ev is None:
                return {"spans_a": [], "spans_b": [], "n_shared": 0,
                        "kind": "moss", "report_url": rep.url,
                        "note": "MOSS không gắn cờ cặp này — xem report gốc"}
            ev["report_url"] = rep.url
            return ev
        return self._evidence_local(code_a, code_b)

    # ----------------------------------------------------------------- local
    def _infer_local(self, code_a: str, code_b: str) -> tuple[float, dict]:
        fa = fingerprints(code_a, k=self.k, w=self.w)
        fb = fingerprints(code_b, k=self.k, w=self.w)
        score = similarity(fa, fb)
        return score, {
            "mode": "moss-local", "n_fp_a": len(fa), "n_fp_b": len(fb),
            "k": self.k, "w": self.w,
        }

    def _evidence_local(self, code_a: str, code_b: str) -> dict | None:
        fa = fingerprints_with_spans(code_a, k=self.k, w=self.w)
        fb = fingerprints_with_spans(code_b, k=self.k, w=self.w)
        sa, sb, n = shared_spans(fa, fb)
        if not n:
            return {"spans_a": [], "spans_b": [], "n_shared": 0, "kind": "kgram",
                    "note": "không có k-gram chung (k=%d, w=%d)" % (self.k, self.w)}
        return {"spans_a": [[s, e] for s, e in sa],
                "spans_b": [[s, e] for s, e in sb],
                "n_shared": n, "kind": "kgram",
                "note": "%d k-gram chung (k=%d, w=%d)" % (n, self.k, self.w)}