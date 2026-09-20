"""M3 — ASTNN (stub INIT): interface giữ chỗ, implement đầy đủ ở phase sau.

Tham chiếu: ASTNN chia AST thành các statement-tree và mã hóa bằng RvNN +
bidirectional GRU. Stub này khóa interface để pipeline không phải đổi.
"""

from src.core.interfaces import CloneDetector, ClonePair


class ASTNNCloneDetector(CloneDetector):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._ready = False  # chuyển True khi có parser + model đầy đủ

    def predict_proba(self, pairs: list[ClonePair]) -> list[float]:
        raise NotImplementedError(
            "ASTNN chưa implement ở INIT — xem reports/ast_cpg_feasibility."
        )
