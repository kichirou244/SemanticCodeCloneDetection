"""M3 — ASTNN. Khác M1/M2 ở chỗ không có tokenizer: phải parse C++ thành cây.

Hai điểm đáng chú ý về hiệu năng:

- `VectorizedRvNN` duyệt cây theo **tầng độ sâu** trên tensor phẳng, nên demo
  dùng lại `collate_trees` + `to_device` y hệt runner EXP-004. Không viết lại
  vòng đệ quy Python: bản gốc chậm hàng trăm lần và RQ1 sẽ so sánh M3 với
  implementation khác M1/M2.
- Parse bằng tree-sitter mất ~0.3 ms/code (đo 3 153 file/s ở report 005), tức
  không đáng kể so với forward pass. Nên demo parse trực tiếp, **không** cần
  `ASTTreeStore` (kho đó tồn tại để tiết kiệm thời gian khi lặp lại hàng trăm
  nghìn lần trên 41k code, ở đây chỉ 2 code/lần).

Cả hai phía được collate chung vào **một batch 2** thay vì hai lần forward: đúng
cách `run_batches` của runner làm, và giữ nguyên padding semantics nên score khớp
với lúc evaluate.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.demo.scorers.base import BaseScorer, resolve_device
from src.models.astnn.model import ASTNNPairClassifier, to_device
from src.representations.ast.extract import collate_trees, compile_tree, extract_tree, get_parser


class AstnnScorer(BaseScorer):
    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        checkpoint: Path,
        hidden_size: int = 200,
        dropout: float = 0.2,
        language: str = "cpp",
        max_code_chars: int = 20000,
        device: str = "auto",
    ):
        super().__init__(model_id, label, results_dir, device)
        self.checkpoint = checkpoint
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.language = language
        self.max_code_chars = max_code_chars
        self._model: ASTNNPairClassifier | None = None
        self._parser = None
        self._device: torch.device | None = None

    def checkpoint_hint(self) -> str:
        return f"{self.checkpoint.name} · hidden={self.hidden_size}"

    def _load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"thiếu checkpoint {self.checkpoint} — chạy "
                "`python -m src.experiments.exp004_astnn --config configs/exp004_astnn.yaml "
                "--mode train` trước"
            )
        device = resolve_device(self.device)
        model = ASTNNPairClassifier(self.hidden_size, self.dropout)
        state = torch.load(self.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.to(device).eval()
        self._model = model
        self._parser = get_parser(self.language)
        self._device = device

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        assert self._model is not None and self._parser is not None
        assert self._device is not None
        trees = [
            compile_tree(
                extract_tree(_cap(code, self.max_code_chars), self._parser, self.language)
            )
            for code in (code_a, code_b)
        ]
        # Hai phía là HAI batch riêng (batch_size=1), y hệt `collate_pair` của
        # EXP-004. Gộp chung một batch rồi lấy phần tử đầu sẽ vô tình chấm
        # (a_đối_với_a, a_đối_với_b) — sai hoàn toàn.
        batch_a = to_device(collate_trees([trees[0]]), self._device)
        batch_b = to_device(collate_trees([trees[1]]), self._device)
        with torch.no_grad():
            logit = self._model(batch_a, batch_b)
            prob = torch.sigmoid(logit).item()
        return prob, {
            "n_stmt_a": int(trees[0]["n_stmt"]),
            "n_stmt_b": int(trees[1]["n_stmt"]),
            "capped": any(len(c) >= self.max_code_chars for c in (code_a, code_b)),
            "device": str(self._device),
        }


def _cap(code: str, max_chars: int) -> str:
    return code if len(code) <= max_chars else code[:max_chars]
