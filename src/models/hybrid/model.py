"""M5 — Hybrid GraphCodeBERT + CPG-GNN (stub INIT).

Fusion mặc định INIT: concat [GraphCodeBERT emb; CPG-GNN emb] → MLP → classifier.
Fusion là configurable (configs/models/hybrid.yaml).
"""

from src.core.interfaces import CloneDetector, ClonePair


class HybridCloneDetector(CloneDetector):
    FUSION = "concat_mlp"  # cấu hình INIT; cross-attention để phase sau

    def __init__(self, config: dict | None = None):
        self.config = config or {"fusion": self.FUSION}

    def predict_proba(self, pairs: list[ClonePair]) -> list[float]:
        raise NotImplementedError(
            "Hybrid chưa implement ở INIT — cần M2 và M4 hoàn chỉnh trước."
        )
