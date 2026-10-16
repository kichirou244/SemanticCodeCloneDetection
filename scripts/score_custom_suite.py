"""Cham bo test custom_suite (tai dung cho cac lan test sau).

Bo test mo ta o `data/custom_suite/pairs.jsonl` (them cap moi = them 1 dong,
khong sua code). Chay tu goc repo::

    python scripts/score_custom_suite.py            # all models theo demo_web.yaml
    python scripts/score_custom_suite.py --models B1 M1
    python scripts/score_custom_suite.py --models B1   # nhanh, khong can GPU

Ket qua ghi ra `workspace/custom_suite_scores.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "custom_suite"
OUT_PATH = ROOT / "workspace" / "custom_suite_scores.json"


def load_pairs() -> list[dict]:
    pairs = []
    with open(DATA_DIR / "pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    codes = {p.name: p.read_text(encoding="utf-8")
             for p in DATA_DIR.glob("*.cpp")}
    for rec in pairs:
        rec["code_a"] = codes[rec["a"]]
        rec["code_b"] = codes[rec["b"]]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="cham bo test custom_suite")
    parser.add_argument("--models", nargs="*", default=None,
                        help="VD: --models B1 M1 (mac dinh: tat ca theo demo_web.yaml)")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    from src.demo.registry import ModelRegistry

    reg = ModelRegistry.from_yaml(ROOT / "configs" / "demo_web.yaml", root=ROOT)
    scorers = reg.selected(args.models)
    print("preload:", reg.preload([s.model_id for s in scorers
                                   if s.model_id != "B1"]))
    out: dict = {}
    for rec in load_pairs():
        row: dict = {"label": rec["label"], "category": rec["category"],
                     "note": rec["note"], "scores": {}}
        for s in scorers:
            r = s.score(rec["code_a"], rec["code_b"])
            row["scores"][s.model_id] = {
                "score": r.score, "threshold": r.threshold,
                "verdict": r.verdict, "ok": r.ok, "error": r.error,
            }
        out[rec["pair_id"]] = row
        cells = {m: (v["score"], v["verdict"] if v["ok"] else "ERR")
                 for m, v in row["scores"].items()}
        print(f"{rec['pair_id']} (GT={'clone' if rec['label'] else 'non-clone'}): {cells}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"da ghi {args.out}")


if __name__ == "__main__":
    main()
