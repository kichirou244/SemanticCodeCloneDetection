from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TransformerPairClassifier(nn.Module):
    def __init__(
        self,
        hf_id: str,
        dropout: float = 0.1,
        encoder: nn.Module | None = None,
        freeze_backbone: bool = False,
        pair_architecture: str = "concat_hA_hB_absdiff_mlp",
    ):
        super().__init__()
        self.hf_id = hf_id
        self.freeze_backbone = freeze_backbone
        self.pair_architecture = pair_architecture
        self.encoder = encoder if encoder is not None else AutoModel.from_pretrained(hf_id)
        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.hidden_size = hidden

        # Tùy biến số chiều vào dựa trên cơ chế tương tác cặp
        # diff_mul / absdiff_mul / hadamard: [|hA - hB|, hA ⊙ hB] -> 2 * hidden
        # concat_hA_hB_absdiff_mlp (mặc định): [hA, hB, |hA - hB|] -> 3 * hidden
        if pair_architecture in ("diff_mul", "absdiff_mul", "diff_prod", "hadamard_diff", "diff_hadamard"):
            in_features = 2 * hidden
        else:
            in_features = 3 * hidden

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "head.0.weight"
        if key in state_dict:
            in_features = state_dict[key].shape[1]
            if in_features != self.head[0].in_features:
                self.head[0] = nn.Linear(in_features, self.hidden_size).to(state_dict[key].device)
                if in_features == 2 * self.hidden_size:
                    self.pair_architecture = "diff_mul"
                elif in_features == 3 * self.hidden_size:
                    self.pair_architecture = "concat_hA_hB_absdiff_mlp"
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def encode_side(self, inputs: dict) -> torch.Tensor:
        return self.encoder(**inputs).last_hidden_state[:, 0, :]

    def forward(self, inputs_a: dict, inputs_b: dict) -> torch.Tensor:
        ha = self.encode_side(inputs_a)
        hb = self.encode_side(inputs_b)
        diff = torch.abs(ha - hb)
        if self.pair_architecture in ("diff_mul", "absdiff_mul", "diff_prod", "hadamard_diff", "diff_hadamard"):
            # Phiên bản tương tác: [|hA - hB|, hA ⊙ hB]
            prod = ha * hb
            pair = torch.cat([diff, prod], dim=-1)
        else:
            # Phiên bản gốc: [hA, hB, |hA - hB|]
            pair = torch.cat([ha, hb, diff], dim=-1)
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
