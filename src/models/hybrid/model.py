"""M5 — Hybrid GraphCodeBERT + CPG-GNN (POJ-104, C++).

Tổng hợp quyết định (DEC-005 + DEC-029 + audit `scripts/audit_m2_m4_complementarity.py`):
  - **Fusion mức embedding** (`concat_mlp`), KHÔNG fusion nhị phân OR/AND — audit cho
    thấy AND/OR đều thua M2 (F1 0,879/0,809 < 0,907) còn score fusion thắng
    (logistic test F1 0,9217). Làm mạnh hơn logistic: hợp không gian biểu diễn rồi
    dạy MLP fusion.
  - `h(code) = [h_text ; h_graph]` với `h_text` = [CLS] GraphCodeBERT, `h_graph` =
    `[mean ; max]` readout CPG-GNN; head giữ đúng DEC-016:
    `[hA ; hB ; |hA − hB|] → MLP → logit`.

Mỗi encoder tái dùng thật từ M2/M4 (không clone):
  - text: `AutoModel` GraphCodeBERT — same encode path như `transformer_pair.py`.
  - graph: `CPGEncoder` từ `src.models.cpg_gnn.model` — same class M4.

Warm start: khuyến nghị khởi tạo encoder từ checkpoint M2 (EXP-003) + M4
(EXP-005) rồi train tiếp — lấy đúng biểu diễn mà audit nhìn thấy, rẻ hơn train
từ đầu. `freeze_encoders=True` thì chỉ train head (≈ bản mạnh hơn của logistic).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.cpg_gnn.model import CPGEncoder

NODE_NONE = "hybrid"


class HybridPairClassifier(nn.Module):
    """M5: Siamese hybrid — text encoder (GraphCodeBERT) + graph encoder (CPG-GNN).

    `text_encoder` / `graph_encoder`: cho phép caller truyền encoder sẵn (test,
    demo) — không truyền thì dựng từ pretrained/scratch.
    """

    def __init__(
        self,
        hf_id: str = "microsoft/graphcodebert-base",
        graph_hidden: int = 128,
        graph_num_layers: int = 3,
        graph_heads: int = 4,
        graph_dropout: float = 0.2,
        head_hidden: int = 1024,
        head_dropout: float = 0.1,
        text_dim: int | None = None,
        text_encoder: nn.Module | None = None,
        graph_encoder: nn.Module | None = None,
        freeze_encoders: bool = False,
    ) -> None:
        super().__init__()
        self.hf_id = hf_id
        self.text_encoder = text_encoder if text_encoder is not None else AutoModel.from_pretrained(hf_id)
        if text_dim is None:
            hidden = getattr(getattr(self.text_encoder, "config", None), "hidden_size", None)
            if hidden is None:
                raise RuntimeError(f"Không xác định được text_dim: cần encoder có `config.hidden_size` "
                                   f"hoặc truyền `text_dim`")
            text_dim = int(hidden)
        self.text_dim = int(text_dim)
        self.graph_encoder = graph_encoder if graph_encoder is not None else CPGEncoder(
            graph_hidden, graph_num_layers, graph_heads, graph_dropout)
        # Khi caller truyền sẵn encoder, suy `hidden` từ nó (tránh lệch dim âm thầm):
        _g_hidden = getattr(getattr(self.graph_encoder, "feat_lin", None),
                            "out_features", None)
        if _g_hidden is not None:
            graph_hidden = int(_g_hidden)
        self.graph_dim = 2 * int(graph_hidden)
        self.enc_dim = self.text_dim + self.graph_dim
        self.freeze_encoders = bool(freeze_encoders)
        if self.freeze_encoders:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            for p in self.graph_encoder.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(head_dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * self.enc_dim, head_hidden),
            nn.Tanh(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.text_encoder.eval()
            self.graph_encoder.eval()
        return self

    def encode_side(self, text_inputs: dict, graph_batch) -> torch.Tensor:
        out = self.text_encoder(**text_inputs)
        h_text = out.last_hidden_state[:, 0, :]
        h_graph = self.graph_encoder(graph_batch)
        return torch.cat([h_text, h_graph], dim=-1)

    def forward(self, text_a: dict, graph_a, text_b: dict, graph_b) -> torch.Tensor:
        h_a = self.encode_side(text_a, graph_a)
        h_b = self.encode_side(text_b, graph_b)
        pair = torch.cat([h_a, h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.head(pair).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def warm_start_text(encoder: nn.Module, path: str, device: torch.device) -> dict:
    """Nạp text encoder từ checkpoint transformers (khóa `encoder.*` của M1/M2).

    Các khóa khác (`head.*`, optimizer, rng) bị bỏ qua. File không tồn tại:
    cảnh báo và tiếp tục (smoke/test chạy không cần artifact).
    """
    _path = __import__("pathlib").Path(path)
    if not _path.exists():
        print(f"[warm-text] bỏ qua: {path} không tồn tại — dùng pretrained")
        return {"loaded": False, "reason": "missing"}
    ckpt = torch.load(_path, map_location=device, weights_only=_path.name.endswith("_weights_only.pt"))
    keys = {k: v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    own = {k.replace("encoder.", "", 1): v for k, v in keys.items()}
    missing, unexpected = encoder.load_state_dict(own, strict=False)
    print(json_dumps({"warm-text": {"loaded_from": path, "layers": len(own),
                                    "missing": len(missing), "unexpected": len(unexpected)}}))
    return {"loaded": True, "layers": len(own), "missing": len(missing),
            "unexpected": len(unexpected)}


def warm_start_graph(encoder: nn.Module, path: str, device: torch.device) -> dict:
    """Nạp graph encoder từ checkpoint M4 (khóa `encoder.*` của CPGGNNPairClassifier)."""
    _path = __import__("pathlib").Path(path)
    if not _path.exists():
        print(f"[warm-graph] bỏ qua: {path} không tồn tại — dùng từ scratch")
        return {"loaded": False, "reason": "missing"}
    ckpt = torch.load(_path, map_location=device, weights_only=_path.name.endswith("_weights_only.pt"))
    keys = {k: v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}
    own = {k.replace("encoder.", "", 1): v for k, v in keys.items()}
    missing, unexpected = encoder.load_state_dict(own, strict=False)
    print(json_dumps({"warm-graph": {"loaded_from": path, "layers": len(own),
                                     "missing": len(missing), "unexpected": len(unexpected)}}))
    return {"loaded": True, "layers": len(own), "missing": len(missing),
            "unexpected": len(unexpected)}


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)