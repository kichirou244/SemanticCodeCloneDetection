"""Giao diện chung để bọc một model thành "chấm điểm 1 cặp code".

Mục đích: `service.py` không cần biết M1 là RoBERTa còn M3 là GRU. Nó chỉ cần
gọi `scorer.score(a, b)` và nhận về `ScoreResult` có cùng hình dạng, kèm ngưỡng
đã đóng băng từ `threshold.json` của chính experiment đó.

Hợp đồng bắt buộc:

- `load()` phải **lazy** và **ghi nhớ kết quả**: gọi `score` lần thứ hai không
  tải lại model. Demo mở trang rồi bấm nhiều lần, tải lại 1.4 GB mỗi lần thì
  không dùng được.
- Lỗi nạp phải được `status()` báo ra chứ không ném ra giữa request, để một
  model hỏng không làm chết cả trang (M2 chỉ có checkpoint đầy đủ 1.44 GB, M4
  thì chưa có gì cả — giao diện vẫn phải chạy được với 3/4 model).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch

#: Ngưỡng chờ nạp model. Quá số này coi như treo, thay vì treo vô hạn khiến
#: frontend quay vô tận.
LOAD_TIMEOUT_S = 300.0


def resolve_device(pref: str) -> torch.device:
    """`auto` → CUDA nếu có, ngược lại CPU.

    Demo chỉ chấm 1 cặp mỗi request nên CPU vẫn đủ nhanh và giúp chạy được trên
    máy không có GPU — không nên fail-fast như các runner nghiên cứu (DEC-018 giữ
    quy tắc đó cho train, không áp cho demo).
    """
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScorerState:
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class ScoreResult:
    """Kết quả chấm một cặp. `ok=False` nghĩa là model lỗi, không phải code lỗi."""

    ok: bool
    score: float | None = None
    threshold: float | None = None
    verdict: str = "unknown"  # "clone" | "not_clone" | "unknown"
    latency_ms: float | None = None
    error: str | None = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "score": self.score,
            "threshold": self.threshold,
            "verdict": self.verdict,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "detail": self.detail,
        }


def read_threshold(results_dir: Path) -> float | None:
    """Đọc ngưỡng đã chọn trên validation từ artifact của experiment.

    Trả `None` nếu chưa eval ⇒ scorer báo `unknown` chứ **không** tự chọn 0.5.
    Chọn nhầm ngưỡng ở demo sẽ ra kết luận sai so với báo cáo, mà người xem demo
    không có cách nào biết mình đang nhìn con số không cùng protocol.
    """
    path = results_dir / "threshold.json"
    if not path.exists():
        return None
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


def read_test_metrics(results_dir: Path) -> dict:
    """Đọc metrics test đã chốt — dùng để hiện uy tín model trên UI."""
    path = results_dir / "metrics.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data.get("test")
    if not test:
        return {}
    return {k: test.get(k) for k in ("precision", "recall", "f1", "roc_auc")}


class BaseScorer(ABC):
    """Một model + ngưỡng của nó, gọi được bằng `score(code_a, code_b)`."""

    def __init__(self, model_id: str, label: str, results_dir: Path, device: str = "auto"):
        self.model_id = model_id
        self.label = label
        self.results_dir = results_dir
        self.device = device
        self._state = ScorerState.NOT_LOADED
        self._error: str | None = None
        self._load_seconds: float | None = None

    # ------------------------------------------------------------- trạng thái
    @property
    def state(self) -> str:
        return self._state

    def status(self) -> dict:
        """Mô tả trạng thái để `GET /api/status` trả về."""
        return {
            "model_id": self.model_id,
            "label": self.label,
            "state": self._state,
            "error": self._error,
            "threshold": read_threshold(self.results_dir),
            "test_metrics": read_test_metrics(self.results_dir),
            "load_seconds": self._load_seconds,
        }

    @abstractmethod
    def _load(self) -> None:
        """Nạp trọng. Chỉ được gọi bởi `ensure_loaded`."""

    @abstractmethod
    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        """Chạy model, trả `(xác suất, chi tiết)`. Chỉ gọi khi đã `ensure_loaded`."""

    def checkpoint_hint(self) -> str:
        return ""

    def evidence(self, code_a: str, code_b: str) -> dict | None:
        """Bằng chứng vị trí cho highlight riêng từng model.

        Mặc định None (chưa có). MOSS override: spans k-gram chung (tọa độ code
        gốc). Service map spans → chỉ số token rồi gắn vào response; frontend
        tô theo model đang chọn. Model neural (M1/M2/M3/M4/M5) làm sau.
        """
        return None

    # ------------------------------------------------------------------ gọi
    def ensure_loaded(self) -> bool:
        """Nạp nếu chưa nạp. Trả `False` nếu hỏng (đã ghi lý do vào `_error`)."""
        if self._state == ScorerState.READY:
            return True
        if self._state == ScorerState.FAILED:
            return False
        self._state = ScorerState.LOADING
        self._error = None
        t0 = time.perf_counter()
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 — lỗi nạp phải thành trạng thái, không thành 500
            self._state = ScorerState.FAILED
            self._error = f"{type(exc).__name__}: {exc}"[:400]
            return False
        self._load_seconds = round(time.perf_counter() - t0, 2)
        self._state = ScorerState.READY
        return True

    def score_raw(self, code_a: str, code_b: str) -> tuple[float | None, dict]:
        """Điểm thô + chi tiết, KHÔNG verdict (service tự tính theo ngưỡng áp
        dụng — ngưỡng frozen hay override ngoài). Lỗi nạp/infer → None + lý do
        trong detail, không ném."""
        r = self.score(code_a, code_b)
        if not r.ok:
            return None, {"error": r.error}
        return r.score, r.detail

    def score_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[float | None, dict]]:
        """Chấm nhiều cặp một lần. Mặc định: vòng per-pair (giữ hành vi cũ).
        Scorer nặng (transformer) override để forward gộp — service chia chunk
        rồi gọi hàm này nên tiến độ vẫn báo đều."""
        return [self.score_raw(a, b) for a, b in items]

    def score(self, code_a: str, code_b: str) -> ScoreResult:
        """Chấm một cặp. Không bao giờ ném — lỗi trả về trong `ScoreResult`."""
        threshold = read_threshold(self.results_dir)
        if not self.ensure_loaded():
            return ScoreResult(ok=False, threshold=threshold, error=self._error,
                               detail={"state": self._state})
        t0 = time.perf_counter()
        try:
            score, detail = self._infer(code_a, code_b)
        except Exception as exc:  # noqa: BLE001 — xem `ensure_loaded`
            return ScoreResult(ok=False, threshold=threshold,
                               error=f"{type(exc).__name__}: {exc}"[:400])
        latency = round((time.perf_counter() - t0) * 1000, 1)
        verdict = "unknown" if threshold is None else (
            "clone" if score >= threshold else "not_clone"
        )
        return ScoreResult(ok=True, score=round(float(score), 6), threshold=threshold,
                           verdict=verdict, latency_ms=latency, detail=detail)
