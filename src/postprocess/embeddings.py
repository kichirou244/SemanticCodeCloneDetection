"""Trích vector CLS (hA, hB) từ checkpoint DiffMul để phân tích phân bố lớp."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.postprocess.inference import build_model_from_checkpoint


@torch.no_grad()
def extract_pair_embeddings(
    checkpoint: dict[str, Any] | str,
    pairs: list[Any],
    batch_size: int = 32,
    max_length: int = 512,
    device: torch.device | None = None,
    use_amp: bool = True,
    label: str = "model",
) -> tuple[np.ndarray, np.ndarray]:
    """Trả (HA, HB) với mỗi array shape (N, hidden). Giữ đúng thứ tự pairs."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(checkpoint, (str,)) or hasattr(checkpoint, "__fspath__"):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model, hf_id = build_model_from_checkpoint(checkpoint, device)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    ha_all: list[list[float]] = []
    hb_all: list[list[float]] = []
    amp_enabled = bool(use_amp and device.type == "cuda")

    print(f"[{label}] extract embeddings: {len(pairs)} pairs | {hf_id} | bs={batch_size}", flush=True)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        codes_a = [str(p.func1).replace("\r\n", "\n").replace("\r", "\n").strip() for p in chunk]
        codes_b = [str(p.func2).replace("\r\n", "\n").replace("\r", "\n").strip() for p in chunk]
        inputs_a = tok(codes_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        inputs_b = tok(codes_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            ha = model.encode_side(inputs_a)
            hb = model.encode_side(inputs_b)

        ha_all.extend(ha.float().cpu().tolist())
        hb_all.extend(hb.float().cpu().tolist())
        done = min(start + batch_size, len(pairs))
        if done % (batch_size * 25) == 0 or done == len(pairs):
            print(f"  [{label}] {done}/{len(pairs)}", flush=True)

    return np.asarray(ha_all, dtype=np.float32), np.asarray(hb_all, dtype=np.float32)
