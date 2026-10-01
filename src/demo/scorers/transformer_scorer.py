"""M1 / M2 — CodeBERT và GraphCodeBERT, dùng chung một `TransformerPairClassifier`.

Hai model này khác nhau đúng ở trọng số pretrained, nên demo dùng **chung** một
scorer, giống hệt `transformer_pair_runner` dùng chung một runner cho EXP-002 và
EXP-003. Nếu tách hai class riêng thì RQ1 so sánh M1 với M2 sẽ lẫn cả khác biệt
implementation vào kết quả — đúng cái lỗi DEC-020 sinh ra để tránh.

Về `max_length = 256`: đây là ngưỡng đã đóng băng (DEC-016). Demo không tăng lên
512 để "cho kết quả đẹp hơn" — tăng sẽ ra số khác số đã báo cáo trong
`metrics.json`, mà chính số đó lại hiện cạnh model trên UI.

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
from src.experiments.transformer_pair_runner import load_eval_weights
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
