"""Lưu trữ false positives / false negatives cho error analysis."""

import json
from pathlib import Path


def save_error(case: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kind = "false_positive" if case.get("prediction") == 1 else "false_negative"
    path = out / kind / f"{case.get('experiment_id', 'exp')}_{case.get('pair_id', 'x')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
