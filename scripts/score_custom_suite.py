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
    parser.add_argument("--only-new", action="store_true",
                        help="chi cham cac cap CHUA co trong file --out; can --merge. "
                             "Dung sau khi them dong vao pairs.jsonl de khong cham lai "
                             "cac cap cu (M4/Joern rat dat)")
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--batch-pairs", type=int, default=16,
                        help="so cap moi lan forward cho M1/M2/M3/M5 (tranh OOM; "
                             "M4 van 1 lan Joern cho toan bo file)")
    args = parser.parse_args()
    if args.only_new and not args.merge:
        parser.error("--only-new phai di kem --merge (neu khong se ghi de mat cap cu)")

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

    out_path = Path(args.out)
    if args.only_new:
        prev = (json.loads(out_path.read_text(encoding="utf-8"))
                if out_path.exists() else {})
        before = len(all_pairs)
        all_pairs = [r for r in all_pairs if r["pair_id"] not in prev]
        print(f"--only-new: {len(all_pairs)}/{before} cap chua co trong {out_path}")
        if not all_pairs:
            print("khong co cap moi de cham")
            return

    reg = ModelRegistry.from_yaml(ROOT / "configs" / "demo_web.yaml", root=ROOT)
    scorers = reg.selected(args.models)
    out: dict = {}
    if args.merge:
        if not out_path.exists():
            parser.error(f"--merge nhung chua co {out_path} de cap nhat")
        out = json.loads(out_path.read_text(encoding="utf-8"))
    print("preload:", reg.preload([s.model_id for s in scorers
                                   if s.model_id != "B1"]))

    # Cham theo lo: M4/M5 chay Joern mot lan cho toan bo file, M1/M2/M3 forward
    # gop. Cham tung cap se khoi dong JVM ~30-60s/cap (M4/M5) — 48 cap la hang
    # gio. `score_all_pairs`/`score_batch` đã có sẵn trong các scorer.
    from src.demo.scorers.base import read_threshold

    results: dict[str, dict[str, tuple[float | None, dict]]] = {}
    for s in scorers:
        mid = s.model_id
        if mid == "B1" or not hasattr(s, "score_batch"):
            # B1 (MOSS service/local) khong co batch; giu nguyen tung cap.
            results[mid] = {
                rec["pair_id"]: s.score_raw(rec["code_a"], rec["code_b"])
                for rec in all_pairs
            }
            continue
        if mid == "M4" and hasattr(s, "score_all_pairs"):
            # M4 chi co score_all_pairs (khong override score_batch).
            names: list[str] = []
            codes: dict[str, str] = {}
            for rec in all_pairs:
                for name, code in ((rec["a"], rec["code_a"]),
                                   (rec["b"], rec["code_b"])):
                    if name not in codes:
                        names.append(name)
                        codes[name] = code
            raw = s.score_all_pairs([(n, codes[n]) for n in names])
            by_key = {frozenset(k): v for k, v in raw.items()}
            results[mid] = {
                rec["pair_id"]: by_key.get(
                    frozenset((rec["a"], rec["b"])),
                    (None, {"error": "khong co ket qua batch cho cap nay"}))
                for rec in all_pairs
            }
            continue
        results[mid] = {}
        step = max(1, int(args.batch_pairs))
        for start in range(0, len(all_pairs), step):
            chunk = all_pairs[start:start + step]
            got = s.score_batch([(rec["code_a"], rec["code_b"]) for rec in chunk])
            results[mid].update({rec["pair_id"]: r for rec, r in zip(chunk, got)})

    fresh: set[str] = set()
    for rec in all_pairs:
        row = out.setdefault(rec["pair_id"], {"scores": {}})
        row["label"] = rec["label"]
        row["category"] = rec["category"]
        row["note"] = rec["note"]
        row.setdefault("scores", {})
        for s in scorers:
            mid = s.model_id
            threshold = read_threshold(s.results_dir)
            score, detail = results[mid].get(
                rec["pair_id"], (None, {"error": "thieu ket qua"}))
            if score is None:
                row["scores"][mid] = {
                    "score": None, "threshold": threshold,
                    "verdict": "unknown", "ok": False,
                    "error": detail.get("error") or "khong cham duoc",
                }
            else:
                score = round(float(score), 6)
                verdict = ("unknown" if threshold is None else
                           ("clone" if score >= threshold else "not_clone"))
                row["scores"][mid] = {
                    "score": score, "threshold": threshold,
                    "verdict": verdict, "ok": True, "error": None,
                }
            fresh.add(mid)
        cells = {m: (v["score"], v["verdict"] if v["ok"] else "ERR")
                 for m, v in row["scores"].items() if m in fresh}
        print(f"{rec['pair_id']} (GT={'clone' if rec['label'] else 'non-clone'}): {cells}")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"da ghi {out_path}")


if __name__ == "__main__":
    main()
