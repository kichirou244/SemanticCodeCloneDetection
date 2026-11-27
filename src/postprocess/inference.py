"""Suy luận logits cho các checkpoint DiffMul (dùng chung cho val/test)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.models.transformer_pair import TransformerPairClassifier


def load_pairs(path: str | Path, limit: int | None = None) -> list[Any]:
    """Nạp cặp từ parquet (bcb_validation.parquet / bcb_test.parquet)."""
    pairs = BigCloneBenchLoader().load_parquet(Path(path), max_samples=limit)
    if not pairs:
        raise RuntimeError(f"Không nạp được cặp nào từ {path}")
    return pairs


def build_model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> tuple[TransformerPairClassifier, Any]:
    """Khởi tạo TransformerPairClassifier theo cfg lưu trong checkpoint."""
    cfg = ckpt.get("cfg", {})
    model_cfg = cfg.get("model", {})
    hf_id = model_cfg.get("hf_id", "microsoft/codebert-base")
    pair_arch = model_cfg.get("pair_architecture", "diff_mul")
    dropout = float(model_cfg.get("dropout", 0.1))

    model = TransformerPairClassifier(hf_id=hf_id, dropout=dropout, pair_architecture=pair_arch).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"  [warn] thiếu {len(missing)} key khi nạp state_dict: {missing[:3]}", flush=True)
    if unexpected:
        print(f"  [warn] thừa {len(unexpected)} key khi nạp state_dict: {unexpected[:3]}", flush=True)
    model.eval()
    return model, hf_id


@torch.no_grad()
def infer_logits(
    checkpoint: dict[str, Any] | str | Path,
    pairs: list[Any],
    batch_size: int = 32,
    max_length: int = 512,
    device: torch.device | None = None,
    use_amp: bool = True,
    label: str = "model",
) -> np.ndarray:
    """Trả về logits (float64 numpy) cho từng cặp, theo đúng thứ tự đầu vào."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model, hf_id = build_model_from_checkpoint(checkpoint, device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    print(f"[{label}] {hf_id} | {len(pairs)} pairs | device={device} | bs={batch_size}", flush=True)

    logits: list[float] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        codes_a = [str(p.func1).replace("\r\n", "\n").replace("\r", "\n").strip() for p in chunk]
        codes_b = [str(p.func2).replace("\r\n", "\n").replace("\r", "\n").strip() for p in chunk]
        inputs_a = tokenizer(
            codes_a, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        inputs_b = tokenizer(
            codes_b, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            ha = model.encode_side(inputs_a)
            hb = model.encode_side(inputs_b)
            arch = getattr(model, "pair_architecture", "diff_mul")
            if arch in ("diff_mul", "absdiff_mul", "diff_prod", "hadamard_diff", "diff_hadamard"):
                pair_repr = torch.cat([torch.abs(ha - hb), ha * hb], dim=-1)
            else:
                pair_repr = torch.cat([ha, hb, torch.abs(ha - hb)], dim=-1)
            out = model.head(pair_repr).squeeze(-1)
        logits.extend(out.float().cpu().tolist())

        done = min(start + batch_size, len(pairs))
        if done % (batch_size * 25) == 0 or done == len(pairs):
            print(f"  [{label}] {done}/{len(pairs)}", flush=True)

    return np.asarray(logits, dtype=np.float64)


def save_cache(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)


def load_cache(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    return np.load(p) if p.exists() else None
