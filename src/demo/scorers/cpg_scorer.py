"""M4 — CPG-GNN. Khác 4 scorer còn lại ở đúng một điểm: input phải qua Joern.

Vì Joern cần JVM (khởi động + parse vài chục giây cho 2 hàm), `_infer` dựng
batch 2 file tạm rồi gọi đúng `run_batch` của `scripts/build_cpg_export` —
không viết lại đường export thứ hai, nên CPG demo thấy là CPG train đã thấy.
Đồ thị đọc bằng `ingest.parse_cpg_file` (đã có test), collate hai phía riêng
bằng `collate_pyg` (gộp chung một batch sẽ chấm nhầm cặp — xem chú thích tương
tự trong `astnn_scorer.py`).

Máy không có Java/Joern (đúng trường hợp máy dự án, DEC-023) thì `_load` báo
hỏng với lý do rõ ràng — trang demo vẫn chạy với 4 model còn lại, M4 hiện
trạng thái lỗi chứ không kéo sập request.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from scripts.build_cpg_export import (
    joern_missing_reason,
    joern_or_die,
    run_batch,
    write_batch,
)
from src.demo.scorers.base import BaseScorer, resolve_device
from src.representations.cpg.ingest import parse_cpg_file

#: Joern cho 2 hàm nhỏ xong trong ~1 phút; quá số này là treo chứ không phải chậm.
JOERN_TIMEOUT_S = 300


class CpgScorer(BaseScorer):
    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        checkpoint: Path,
        hidden_size: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        max_nodes: int = 1500,
        device: str = "auto",
    ):
        super().__init__(model_id, label, results_dir, device)
        self.checkpoint = checkpoint
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.max_nodes = max_nodes
        self._model = None
        self._device = None
        self._joern_cmd = ""

    def checkpoint_hint(self) -> str:
        return f"{self.checkpoint.name} · gat{self.num_layers}x{self.heads}"

    def _load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"thiếu checkpoint {self.checkpoint} — chạy notebook train M4 "
                "(bước 2) trước"
            )
        # Kiểm tra môi trường TRƯỚC import nặng: mỗi máy thiếu thứ khác nhau
        # (máy này thiếu Java, máy khác có thể thiếu torch_geometric) và thông
        # báo phải chỉ đúng thứ thiếu — fail fast thay vì lỗi import khó đọc.
        joern_cmd = joern_or_die()
        missing = joern_missing_reason(joern_cmd)
        if missing:
            raise RuntimeError(
                f"{missing}. M4 cần Java + Joern để dựng CPG lúc chấm điểm "
                "(xem notebooks/exp005_cpg_extract_kaggle.ipynb)"
            )
        # Import nặng để trong `_load` (lazy): máy không có torch_geometric
        # vẫn import được registry/server, M4 chỉ báo trạng thái lỗi.
        import torch

        from src.models.cpg_gnn.model import CPGGNNPairClassifier

        device = resolve_device(self.device)
        model = CPGGNNPairClassifier(self.hidden_size, self.num_layers,
                                     self.heads, self.dropout)
        state = torch.load(self.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        self._model = model
        self._device = device
        self._joern_cmd = joern_cmd

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        assert self._model is not None and self._device is not None
        import torch

        from src.models.cpg_gnn.model import to_device
        from src.representations.cpg.graph import collate_pyg

        samples = [
            {"code": code_a, "sample_id": "demo_a"},
            {"code": code_b, "sample_id": "demo_b"},
        ]
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="cpg_demo_") as tmp:
            work = Path(tmp)
            batch_dir = work / "batch"
            out_dir = work / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            write_batch(samples, batch_dir)
            n_ok, _ = run_batch(samples, out_dir, work, self._joern_cmd,
                                batch_dir, JOERN_TIMEOUT_S)
            if n_ok < 2:
                raise RuntimeError(
                    "Joern không ra JSON cho cả 2 phía — xem log server "
                    "(thường là lỗi compile Scala hoặc thiếu Java)")
            graphs = []
            for sid in ("demo_a", "demo_b"):
                g, _ = parse_cpg_file(out_dir / f"{sid}.json", self.max_nodes)
                graphs.append(g)
        joern_ms = round((time.perf_counter() - t0) * 1000, 1)
        batch_a = to_device(collate_pyg([graphs[0]]), self._device)
        batch_b = to_device(collate_pyg([graphs[1]]), self._device)
        with torch.no_grad():
            logit = self._model(batch_a, batch_b)
            prob = torch.sigmoid(logit).item()
        return prob, {
            "n_nodes_a": int(graphs[0].n_nodes),
            "n_nodes_b": int(graphs[1].n_nodes),
            "joern_ms": joern_ms,
            "device": str(self._device),
        }
