from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerPairClassifier(nn.Module):
    def __init__(self, hf_id: str, dropout: float = 0.1, encoder: nn.Module | None = None, freeze_backbone: bool = False):
        super().__init__()
        self.hf_id = hf_id
        self.freeze_backbone = freeze_backbone
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(hf_id)
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.hidden_size = hidden

    def encode_side(self, inputs: dict) -> torch.Tensor:
        return self.encoder(**inputs).last_hidden_state[:, 0, :]

    def forward(self, inputs_a: dict, inputs_b: dict) -> torch.Tensor:
        ha = self.encode_side(inputs_a)
        hb = self.encode_side(inputs_b)
        pair = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
        return self.head(pair).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_tokenizer(hf_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(hf_id)


def tokenize_side(tokenizer: AutoTokenizer, codes: list[str], max_length: int) -> tuple[dict, int]:
    enc = tokenizer(
        codes,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = {k: v for k, v in enc.items()}
    actual = int(enc["attention_mask"].sum(dim=1).max().item()) if "attention_mask" in enc else max_length
    return tokens, min(actual, max_length)
