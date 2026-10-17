"""Cham bo test custom_suite (tai dung cho cac lan test sau).

Bo test mo ta o `data/custom_suite/pairs.jsonl` (them cap moi = them 1 dong,
khong sua code). Chay tu goc repo::

    python scripts/score_custom_suite.py            # all models theo demo_web.yaml
    python scripts/score_custom_suite.py --models B1 M1
    python scripts/score_custom_suite.py --models B1   # nhanh, khong can GPU
    python scripts/score_custom_suite.py --models M1 M2 M5 --merge   # cham lai 3 model
                                                                     # 512, giu B1/M3/M4
    python scripts/score_custom_suite.py --pairs P5_struct_class --models M1 M2 M5 --merge
        # chi cham cap moi (nhom P5), giu nguyen 26 cap cu

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
    parser.add_argument("--merge", action="store_true",
                        help="chi cap nhat cac model trong --models, giu nguyen "
                             "diem da co cua cac model khac (VD chấm lai M1/M2/M5 "
                             "ma khong cham B1 khi may thieu MOSS_USER_ID)")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="chi cham cac pair_id nay (mac dinh: tat ca trong "
                             "pairs.jsonl), dung voi --merge de cham rieng cap moi")
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    from src.demo.registry import ModelRegistry

    all_pairs = load_pairs()
    if args.pairs:
        wanted = set(args.pairs)
        known = {r["pair_id"] for r in all_pairs}
        missing = sorted(wanted - known)
        if missing:
            parser.error(f"--pairs khong co trong pairs.jsonl: {missing}")
        all_pairs = [r for r in all_pairs if r["pair_id"] in wanted]
        print(f"chi cham {len(all_pairs)} cap: {[r['pair_id'] for r in all_pairs]}")

    reg = ModelRegistry.from_yaml(ROOT / "configs" / "demo_web.yaml", root=ROOT)
    scorers = reg.selected(args.models)
    out_path = Path(args.out)
    out: dict = {}
    if args.merge:
        if not out_path.exists():
            parser.error(f"--merge nhung chua co {out_path} de cap nhat")
        out = json.loads(out_path.read_text(encoding="utf-8"))
    print("preload:", reg.preload([s.model_id for s in scorers
                                   if s.model_id != "B1"]))
    fresh: set[str] = set()
    for rec in all_pairs:
        row = out.setdefault(rec["pair_id"], {"scores": {}})
        row["label"] = rec["label"]
        row["category"] = rec["category"]
        row["note"] = rec["note"]
        row.setdefault("scores", {})
        for s in scorers:
            r = s.score(rec["code_a"], rec["code_b"])
            row["scores"][s.model_id] = {
                "score": r.score, "threshold": r.threshold,
                "verdict": r.verdict, "ok": r.ok, "error": r.error,
            }
            fresh.add(s.model_id)
        cells = {m: (v["score"], v["verdict"] if v["ok"] else "ERR")
                 for m, v in row["scores"].items() if m in fresh}
        print(f"{rec['pair_id']} (GT={'clone' if rec['label'] else 'non-clone'}): {cells}")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"da ghi {out_path}")


if __name__ == "__main__":
    main()
