"""M1/M2 — Wrapper suy luận embedding cho CodeBERT / GraphCodeBERT.

Smoke test INIT chỉ xác nhận: load model → tokenize → forward → embedding.
Huấn luyện classifier đầy đủ thuộc phase sau.
"""

import torch
from transformers import AutoModel, AutoTokenizer


class TransformerEmbedder:
    MODEL_IDS = {
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
    }

    def __init__(self, method: str = "codebert", device: str | None = None):
        if method not in self.MODEL_IDS:
            raise ValueError(f"method phải thuộc {list(self.MODEL_IDS)}")
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_IDS[method])
        self.model = AutoModel.from_pretrained(self.MODEL_IDS[method]).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, codes: list[str], max_length: int = 512) -> torch.Tensor:
        batch = self.tokenizer(
            codes, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch)
        return out.last_hidden_state[:, 0, :].cpu()  # [CLS] embedding
