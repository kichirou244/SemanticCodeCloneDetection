"""M4 — CPG-GNN (stub INIT): backbone mặc định GAT, cấu hình trong configs/.

Quyết định backbone chính (GGNN vs GAT) được chốt sau feasibility và ghi
vào docs/decision_log.md (DEC). Stub này khóa interface trước.
"""

from src.core.interfaces import CloneDetector, ClonePair


class CPGGNNCloneDetector(CloneDetector):
    BACKBONE = "gat"  # cấu hình chính mặc định; đổi phải kèm decision log

    def __init__(self, config: dict | None = None):
        self.config = config or {"backbone": self.BACKBONE}

    def predict_proba(self, pairs: list[ClonePair]) -> list[float]:
        raise NotImplementedError(
            "CPG-GNN chưa implement ở INIT — xem reports/ast_cpg_feasibility."
        )
