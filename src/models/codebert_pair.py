"""M1 — CodeBERT pair classifier cho binary clone classification (EXP-002).

Architecture (khóa cho EXP-002, DEC-016):
  code_a ──┐
           ├── CodeBERT (shared siamese) → hA / hB ([CLS])
  code_b ──┘
  pair = [hA ; hB ; |hA - hB|] → MLP → logit → P(clone)

- Encoder khởi tạo từ pretrained checkpoint; download fail → exception
  (fail-fast, không fallback random).
- Tokenizer = checkpoint tokenizer (không dùng regex TF-IDF).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CodeBERTPairClassifier(nn.Module):
    def __init__(self, hf_id: str, dropout: float = 0.1):
        super().__init__()
        self.hf_id = hf_id
        # Download lỗi → transformers raise → fail-fast (§15).
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size
        if hidden is None:
            raise RuntimeError(f"Checkpoint {hf_id} thiếu hidden_size (§15)")
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.hidden_size = hidden

    def encode_side(self, inputs: dict) -> torch.Tensor:
        out = self.encoder(**inputs)
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        inputs_a: dict,
        inputs_b: dict,
    ) -> torch.Tensor:
        h_a = self.encode_side(inputs_a)
        h_b = self.encode_side(inputs_b)
        pair = torch.cat([h_a, h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.head(pair).squeeze(-1)  # logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_tokenizer(hf_id: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(hf_id)


def tokenize_side(
    tokenizer: AutoTokenizer,
    codes: list[str],
    max_length: int,
) -> tuple[dict, int]:
    """Tokenize một phía với dynamic padding. Trả về (batch, n_truncated)."""
    plain = tokenizer(
        codes, padding=False, truncation=False, add_special_tokens=True,
    )
    n_truncated = sum(1 for ids in plain["input_ids"] if len(ids) > max_length)
    batch = tokenizer(
        codes, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    return batch, n_truncated


def collate_pair_batch(
    tokenizer: AutoTokenizer,
    codes_a: list[str],
    codes_b: list[str],
    labels: list[int],
    max_length: int,
    device: torch.device,
) -> tuple[dict, dict, torch.Tensor, int]:
    ba, ta = tokenize_side(tokenizer, codes_a, max_length)
    bb, tb = tokenize_side(tokenizer, codes_b, max_length)
    return (
        {k: v.to(device) for k, v in ba.items()},
        {k: v.to(device) for k, v in bb.items()},
        torch.tensor(labels, dtype=torch.float32, device=device),
        ta + tb,
    )
