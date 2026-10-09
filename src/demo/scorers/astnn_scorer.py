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

    def score_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[float | None, dict]]:
        """Parse từng cặp (rẻ, ~ms) rồi forward GỘP 2 phía một lần — đúng cách
        `run_batches` của runner làm. Giữ thứ tự để tách logit theo cặp."""
        if not self.ensure_loaded():
            return [(None, {"error": self._error}) for _ in items]
        assert self._model is not None and self._parser is not None
        assert self._device is not None
        trees_a, trees_b, capped = [], [], False
        for code_a, code_b in items:
            ta = compile_tree(
                extract_tree(_cap(code_a, self.max_code_chars), self._parser, self.language))
            tb = compile_tree(
                extract_tree(_cap(code_b, self.max_code_chars), self._parser, self.language))
            trees_a.append(ta)
            trees_b.append(tb)
            capped = capped or (len(code_a) >= self.max_code_chars
                                or len(code_b) >= self.max_code_chars)
        batch_a = to_device(collate_trees(trees_a), self._device)
        batch_b = to_device(collate_trees(trees_b), self._device)
        with torch.no_grad():
            logits = self._model(batch_a, batch_b)
            probs = torch.sigmoid(logits).cpu().tolist()
        return [(float(p), {
            "n_stmt_a": int(trees_a[i]["n_stmt"]),
            "n_stmt_b": int(trees_b[i]["n_stmt"]),
            "capped": capped,
            "device": str(self._device),
            "batched": True,
        }) for i, p in enumerate(probs)]

    def evidence(self, code_a: str, code_b: str) -> dict | None:
        """Cặp câu lệnh giống nhau nhất LẪN NHAU (cosine trên vector RvNN).

        Mỗi statement i phía A tìm bạn giống nhất phía B và ngược lại; chỉ giữ
        cặp hai chiều đều chọn nhau (mutual best) — không ngưỡng số nào, không
        ngẫu nhiên. Bỏ gốc (index 0, span cả hàm — highlight vô nghĩa) và cặp
        cosine <= 0. Vector RvNN (chưa qua GRU) để không phải mổ model.
        """
        if not self.ensure_loaded():
            return None
        assert self._model is not None and self._parser is not None
        assert self._device is not None
        import torch

        from src.models.astnn.model import to_device

        codes = (code_a, code_b)
        trees = []
        for code in codes:
            trees.append(extract_tree(_cap(code, self.max_code_chars),
                                      self._parser, self.language))
        batches = [to_device(collate_trees([compile_tree(t)]), self._device)
                   for t in trees]
        with torch.no_grad():
            vecs = [self._model.rvnn(b) for b in batches]
        # mask statement (bỏ lá): thứ tự DFS của extract khớp đúng thứ tự hàng
        # compiled (stmt_nodes giữ nguyên thứ tự) nên spans lọc theo mask này.
        spans = []
        for t in trees:
            mask = (t["is_leaf"] == 0)
            spans.append([(int(s), int(e)) for s, e in zip(
                t["span_start"][mask], t["span_end"][mask])])
        pairs = []
        na = _unit_rows(vecs[0][1:]) if vecs[0].shape[0] > 1 else vecs[0][:0]
        nb = _unit_rows(vecs[1][1:]) if vecs[1].shape[0] > 1 else vecs[1][:0]
        if na.shape[0] and nb.shape[0]:
            cos = na @ nb.T
            best_b = cos.argmax(dim=1)
            best_a = cos.argmax(dim=0)
            for i in range(cos.shape[0]):
                j = int(best_b[i])
                if int(best_a[j]) == i and float(cos[i, j]) > 0:
                    pairs.append((i + 1, j + 1))  # +1 vì đã bỏ gốc
        out_a = [_byte_span_to_char(codes[0], spans[0][i]) for i, _ in pairs]
        out_b = [_byte_span_to_char(codes[1], spans[1][j]) for _, j in pairs]
        return {"spans_a": out_a, "spans_b": out_b, "n_shared": len(pairs),
                "kind": "stmt",
                "pairs": [{"a": a, "b": b} for a, b in zip(out_a, out_b)],
                "note": "cặp câu lệnh giống nhau nhất lẫn nhau (cosine RvNN)"}

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


def _unit_rows(h):
    """Chuẩn hoá từng hàng về unit vector (hàng 0 giữ 0 → không bao giờ khớp)."""
    import torch

    return h.float() / h.float().norm(dim=1, keepdim=True).clamp(min=1e-9)


def _byte_span_to_char(code: str, span: tuple[int, int]) -> list[int]:
    """Đổi span byte (tree-sitter) sang span ký tự (Python str).

    Biên node luôn rơi vào biên ký tự hợp lệ nên decode prefix là chính xác;
    `errors="ignore"` chỉ là lưới an toàn (không bao giờ kích hoạt đúng ra).
    """
    raw = code.encode("utf-8")
    s, e = span
    cs = len(raw[:max(s, 0)].decode("utf-8", errors="ignore"))
    ce = len(raw[:max(e, 0)].decode("utf-8", errors="ignore"))
    return [cs, ce]
