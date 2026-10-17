"""M1 / M2 — CodeBERT và GraphCodeBERT, dùng chung một `TransformerPairClassifier`.

Hai model này khác nhau đúng ở trọng số pretrained, nên demo dùng **chung** một
scorer, giống hệt `transformer_pair_runner` dùng chung một runner cho EXP-002 và
EXP-003. Nếu tách hai class riêng thì RQ1 so sánh M1 với M2 sẽ lẫn cả khác biệt
implementation vào kết quả — đúng cái lỗi DEC-020 sinh ra để tránh.

Về `max_length`: **phải bằng đúng `max_length` của run mà scorer trỏ tới**, không
được tự ý nâng/hạ. Demo dùng bản 512 của M1/M2/M5 (DEC-031), nên `demo_web.yaml`
khai `max_length: 512`; nếu đổi số này thì điểm hiện cạnh model trên UI sẽ khác
số trong `metrics.json` của chính run đó.

Về cách dựng encoder: ưu tiên `AutoModel.from_config` để **không tải 500 MB
pretrained weights** — demo chỉ cần kiến trúc, trọng số đến từ checkpoint của ta.
`load_state_dict` sẽ fail loud nếu cấu trúc lệch, nên rủi ro là an toàn. Nếu
`from_config` không dùng được (mạng lỗi khi lấy config) thì mới fallback sang
`from_pretrained`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.demo.scorers.base import BaseScorer, resolve_device
from src.experiments.transformer_pair_runner import _assert_device_match, load_eval_weights
from src.models.transformer_pair import TransformerPairClassifier, load_tokenizer


class TransformerScorer(BaseScorer):
    """Bọc một checkpoint transformer theo pair head DEC-016."""

    def __init__(
        self,
        model_id: str,
        label: str,
        results_dir: Path,
        checkpoint: Path,
        hf_id: str,
        max_length: int = 256,
        device: str = "auto",
    ):
        super().__init__(model_id, label, results_dir, device)
        self.checkpoint = checkpoint
        self.hf_id = hf_id
        self.max_length = max_length
        self._model: TransformerPairClassifier | None = None
        self._tokenizer = None
        self._device: torch.device | None = None

    def checkpoint_hint(self) -> str:
        return f"{self.hf_id} · {self.checkpoint.name} · max_length={self.max_length}"

    def _load(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"thiếu checkpoint {self.checkpoint} — chạy "
                f"`python -m src.experiments.{'exp003_graphcodebert' if self.model_id == 'M2' else 'exp002_codebert'} "
                f"--config ... --mode train` trước"
            )
        device = resolve_device(self.device)
        model = TransformerPairClassifier(self.hf_id, encoder=_build_encoder(self.hf_id))
        model.load_state_dict(load_eval_weights(self.checkpoint, device))
        model.to(device).eval()
        self._model = model
        self._tokenizer = load_tokenizer(self.hf_id)
        self._device = device

    def score_batch(
        self, items: list[tuple[str, str]]
    ) -> list[tuple[float | None, dict]]:
        """Forward gộp N cặp: tokenize 2N code một lần, 2 forward (2 phía).
        Padding thêm (pad tới max của 2N thay vì của 2) bị mask che nên toán
        học đồng nhất với per-pair — có test tương đương bắt buộc.
        """
        if not self.ensure_loaded():
            return [(None, {"error": self._error}) for _ in items]
        assert self._model is not None and self._tokenizer is not None
        assert self._device is not None
        texts = [t for ab in items for t in ab]
        with torch.no_grad():
            batch = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            input_a = {k: v[0::2].to(self._device) for k, v in batch.items()}
            input_b = {k: v[1::2].to(self._device) for k, v in batch.items()}
            if self._device.type == "cuda":
                for name, d in (("input_a", input_a), ("input_b", input_b)):
                    for k, v in d.items():
                        if torch.is_tensor(v):
                            _assert_device_match(self._device, v, name=name)
            logits = self._model(input_a, input_b)
            probs = torch.sigmoid(logits).cpu().tolist()
            lengths = batch["attention_mask"].sum(dim=1).tolist()
        out = []
        for i in range(len(items)):
            n_a, n_b = int(lengths[2 * i]), int(lengths[2 * i + 1])
            out.append((float(probs[i]), {
                "n_tokens": max(n_a, n_b),
                "truncated": bool(n_a >= self.max_length or n_b >= self.max_length),
                "device": str(self._device),
                "batched": True,
            }))
        return out

    def evidence(self, code_a: str, code_b: str) -> dict | None:
        """Token ảnh hưởng mạnh tới logit (gradient × input, mỗi phía riêng).

        Forward cặp như lúc chấm (không đụng trọng số), backward một lần rồi
        lấy độ lớn gradient tại embedding mỗi token: token nào đổi là logit đổi
        nhiều thì quan trọng với quyết định. Giữ token trên trung bình, gom
        subword theo từ. Đây là "model dựa vào đâu để quyết", KHÔNG phải
        "đoạn nào khớp nhau" — note ghi rõ để UI không nói quá.

        Vì sao không đọc attention: backend mặc định (`sdpa`) không trả
        attention; ép eager thì tốn thêm cả mô hình trong RAM (đã OOM chết
        server một lần). Saliency chạy được mọi backend, tốn thêm đúng một
        backward transient.
        """
        if not self.ensure_loaded():
            return None
        assert self._model is not None and self._tokenizer is not None
        assert self._device is not None
        model = self._model
        was_training = model.training
        model.eval()
        # Đóng grad trọng số (chỉ cần grad tại embedding): vừa nhẹ RAM
        # (~500MB grad trọng số khỏi cần), vừa không làm bẩn .grad của model.
        frozen = [p for p in model.parameters() if p.requires_grad]
        try:
            for p in frozen:
                p.requires_grad_(False)
            enc = self._tokenizer(
                [code_a, code_b], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping").tolist()
            word_ids = [enc.word_ids(batch_index=i) for i in range(2)]
            inputs = {k: v.to(self._device) for k, v in enc.items()}
            with torch.enable_grad():
                # Detach + requires_grad: graph mới rooted tại embedding (trọng
                # số freeze nên backward CHỈ tích vào đây, không phình 500MB grad
                # trọng số). Mirror đúng forward() (eval nên dropout là identity).
                emb = model.encoder.embeddings(input_ids=inputs["input_ids"])
                emb = emb.detach().requires_grad_(True)
                out = model.encoder(inputs_embeds=emb,
                                    attention_mask=inputs["attention_mask"])
                cls = out.last_hidden_state[:, 0, :]
                ha, hb = cls[0:1], cls[1:2]
                pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
                logit = model.head(pair).squeeze(-1)
                logit.backward()
                grad = emb.grad
                assert grad is not None
                sal = (emb.detach() * grad.detach()).abs().sum(-1).cpu()
            out_spans = []
            for side in range(2):
                by_word: dict[int, list] = {}
                off, wid = offsets[side], word_ids[side]
                for i, (w, (s, e)) in enumerate(zip(wid, off)):
                    if w is None or e <= s:
                        continue
                    by_word.setdefault(w, []).append(
                        (float(sal[side, i]), s, e))
                scored = [(max(w for w, _, _ in v),
                           min(s for _, s, _ in v), max(e for _, _, e in v))
                          for v in by_word.values()]
                if not scored:
                    out_spans.append([])
                    continue
                mean = sum(w for w, _, _ in scored) / len(scored)
                out_spans.append([[s, e] for w, s, e in scored if w > mean])
        finally:
            for p in frozen:
                p.requires_grad_(True)
            model.train(was_training)
        # Tương ứng từng từ: mỗi từ phía A tìm từ giống nhất phía B trên
        # vector encoder của chính model, giữ cặp hai chiều đều chọn nhau.
        # Đây mới là "đoạn nào khớp đoạn nào" theo model, khác saliency ở trên
        # vốn chỉ nói "model dựa vào đâu để cho điểm".
        pairs = self._word_pairs(code_a, code_b)
        return {"spans_a": out_spans[0], "spans_b": out_spans[1],
                "n_shared": len(out_spans[0]) + len(out_spans[1]),
                "kind": "saliency",
                "pairs": pairs,
                "note": "token ảnh hưởng mạnh tới logit (mỗi phía riêng)"}

    def _word_pairs(self, code_a: str, code_b: str) -> list[dict]:
        """Cặp từ tương ứng mutual-best trên vector encoder, tối đa 30 cặp."""
        import torch

        from src.demo.correspondence import mutual_best, word_vectors

        assert self._model is not None and self._tokenizer is not None
        assert self._device is not None
        model = self._model
        was_training = model.training
        model.eval()
        try:
            enc = self._tokenizer(
                [code_a, code_b], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping").tolist()
            word_ids = [enc.word_ids(batch_index=i) for i in range(2)]
            inputs = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                h = model.encoder(**inputs).last_hidden_state.cpu()
        finally:
            model.train(was_training)
        va, sa = word_vectors(h[0], word_ids[0], offsets[0])
        vb, sb = word_vectors(h[1], word_ids[1], offsets[1])
        return [{"a": sa[i], "b": sb[j], "cos": round(c, 4)}
                for i, j, c in mutual_best(va, vb)]

    def _infer(self, code_a: str, code_b: str) -> tuple[float, dict]:
        assert self._model is not None and self._tokenizer is not None
        assert self._device is not None
        with torch.no_grad():
            batch = self._tokenizer(
                [code_a, code_b], padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            tokens = batch["input_ids"].shape[1]
            input_a = {k: v[:1].to(self._device) for k, v in batch.items()}
            input_b = {k: v[1:].to(self._device) for k, v in batch.items()}
            logit = self._model(input_a, input_b)
            prob = torch.sigmoid(logit).item()
        return prob, {
            "n_tokens": int(tokens),
            "truncated": bool(tokens >= self.max_length),
            "device": str(self._device),
        }


def _build_encoder(hf_id: str):
    """Dựng encoder từ config (không tải trọng số), fallback `from_pretrained`."""
    from transformers import AutoConfig, AutoModel

    try:
        return AutoModel.from_config(AutoConfig.from_pretrained(hf_id))
    except Exception:  # noqa: BLE001 — mất mạng thì thử đường tải đầy đủ
        return AutoModel.from_pretrained(hf_id)
