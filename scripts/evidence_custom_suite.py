"""Sinh evidence LaTeX cho cac cap custom_suite (tai dung cho bao cao).

Chay tu goc repo::

    python scripts/evidence_custom_suite.py                        # 2 cap mac dinh
    python scripts/evidence_custom_suite.py --pairs P1_combo       # 1 cap
    python scripts/evidence_custom_suite.py --pairs P1_combo P3_nonclone_sum X_P2P3
    python scripts/evidence_custom_suite.py --reuse-moss --pairs ...   # giu artifact MOSS that
    python scripts/evidence_custom_suite.py --pairs P5_struct_class --models M2 M5
        # code dai (70 dong): chi ve hinh 2 model cho gon bao cao

Ra `workspace/demo_suite_<pair>.tex` (5 model M1..M5, giong demo_clone5.tex)
+ `workspace/demo_suite_moss_<pair>.tex` (MOSS server, giong demo_moss_61.tex)
+ `workspace/demo_suite_evidence.json` (diem + evidence tho).

Ton vai phut (nap M1/M2/M3/M4/M5, M1/M2 can backward cho saliency, M4/M5 can
Joern). B1 lay evidence tu trang match Stanford (can mang + mosspy).

`--reuse-moss`: giu nguyen `demo_suite_moss_<pair>.tex` va khoi B1 da co trong
JSON thay vi cham lai. Dung khi may khong khai `MOSS_USER_ID` — luc do B1 roi vao
fallback winnowing local, va ghi de se lam mat artifact MOSS that cua bao cao.

Code dai (>48 dong, vi du nhom P5 DSLK/OOP) tu dong dung `\\scriptsize` cho
minipage de hinh con vua mot trang A4; code qua dai (>90 dong, nhu nhom P6
127-149 dong) dung `\\tiny` va chi hien FIG_MAX_LINES dau tien, phan con lai
danh dau bang dong `\\ldots` (`fig_layout`).

`workspace/demo_suite_evidence.json` mang tinh tich luy: cac cap da co tu cac
lan truoc duoc giu nguyen, chi cac cap chay trong lan nay bi ghi de.
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
OUT_JSON = ROOT / "workspace" / "demo_suite_evidence.json"

#: Thu tu + tieu de nhu bao cao cu (demo_clone5.tex).
MODEL_ORDER = ["M1", "M2", "M3", "M4", "M5"]
MODEL_TITLES = {
    "M1": "CodeBERT",
    "M2": "GraphCodeBERT",
    "M3": "ASTNN",
    "M4": "CPG-GNN",
    "M5": "Hybrid",
}
VERDICT_VI = {"clone": "clone", "not_clone": "không clone", "unknown": "?"}


def fmt_score(score: float | None) -> str:
    return "---" if score is None else f"{score:.3f}".replace(".", ",")


#: Cách đặt chữ + số dòng hiển thị cho một cặp, theo độ dài code (xem fig_layout).
FIG_FULL_LINES = 48
#: Code dài hơn ngưỡng này thì hai cột A|B không còn vừa một trang A4 ở bất kỳ
#: cỡ chữ nào (nhóm P6: 126–148 dòng; \\tiny + fboxsep 1pt cũng ~7pt/dòng
#: ⇒ hơn 700pt). Lúc đó chỉ hiện đầu file và đánh dấu phần còn lại bằng \\ldots.
FIG_MAX_LINES = 90


def fig_layout(code_a: str, code_b: str) -> tuple[str, int | None]:
    """(cỡ chữ, số dòng hiển thị tối đa) cho minipage theo độ dài code.

    File dài (>48 dòng) dùng \\scriptsize (nhóm P5, 59–70 dòng); file quá dài
    (>90 dòng, như nhóm P6 126–148 dòng) dùng \\tiny + giới hạn hiển thị
    FIG_MAX_LINES dòng đầu để hình vẫn vừa một trang A4.
    """
    n = max(code_a.count("\n") + 1, code_b.count("\n") + 1)
    if n > FIG_MAX_LINES:
        return "\\tiny", FIG_MAX_LINES
    if n > FIG_FULL_LINES:
        return "\\scriptsize", None
    return "\\footnotesize", None


def fig_font(code_a: str, code_b: str) -> str:
    """Cỡ chữ minipage theo độ dài code (chỉ cỡ chữ; xem fig_layout)."""
    return fig_layout(code_a, code_b)[0]


def load_suite() -> tuple[dict, dict]:
    pairs = {}
    with open(DATA_DIR / "pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pairs[r["pair_id"]] = r
    codes = {p.name: p.read_text(encoding="utf-8")
             for p in DATA_DIR.glob("*.cpp")}
    return pairs, codes


def render_model_section(
    latex_parts, mid: str, title: str, code_a: str, code_b: str,
    score, verdict: str, ev: dict | None,
    title_a: str = "File A", title_b: str = "File B",
    font: str = "\\footnotesize",
    max_lines: int | None = None,
) -> None:
    """Mot model -> subsubsection + 2 minipage (giong demo_clone5.tex)."""
    from scripts.scan_demo_zip import _latex_minipage

    tok_a: set[int] = set((ev or {}).get("tok_a", []))
    tok_b: set[int] = set((ev or {}).get("tok_b", []))
    pair_of_a: dict[int, int] = {}
    pair_of_b: dict[int, int] = {}
    for pi, p in enumerate((ev or {}).get("tok_pairs", [])):
        for i in p.get("a", []):
            pair_of_a.setdefault(i, pi)
        for i in p.get("b", []):
            pair_of_b.setdefault(i, pi)
    kind = (ev or {}).get("kind")
    note = (ev or {}).get("note")
    latex_parts.append(f"\\subsubsection*{{{title} --- {fmt_score(score)} ({verdict})}}")
    if note:
        latex_parts.append(f"% kind={kind}; {note}")
    latex_parts += _latex_minipage(code_a, tok_a, title_a, pair_of_a, font,
                                   max_lines=max_lines)
    latex_parts.append("\\hfill")
    latex_parts += _latex_minipage(code_b, tok_b, title_b, pair_of_b, font,
                                   max_lines=max_lines)
    latex_parts.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description="evidence LaTeX custom_suite")
    parser.add_argument("--pairs", nargs="*", default=["P1_combo", "P3_nonclone_sum"],
                        help="pair_id trong data/custom_suite/pairs.jsonl")
    parser.add_argument("--reuse-moss", action="store_true",
                        help="giu nguyen demo_suite_moss_<pair>.tex va khoi B1 da co "
                             "trong JSON thay vi cham lai (dung khi may khong khai "
                             "MOSS_USER_ID, tranh ghi de bang fallback local)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="chi ve hinh cac model nay trong demo_suite_<pair>.tex "
                             "(mac dinh M1..M5). JSON van luu du 5 model; B1/MOSS "
                             "luon ghi file rieng")
    args = parser.parse_args()

    render_models = args.models or MODEL_ORDER
    bad = [m for m in render_models if m not in MODEL_ORDER]
    if bad:
        parser.error(f"--models chi nhan {MODEL_ORDER}, khong co: {bad}")

    from src.demo.registry import ModelRegistry
    from src.demo.service import DemoService

    old: dict = {}
    if args.reuse_moss:
        if not OUT_JSON.exists():
            parser.error(f"--reuse-moss nhung chua co {OUT_JSON} de giu lai")
        old = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    pairs, codes = load_suite()
    reg = ModelRegistry.from_yaml(ROOT / "configs" / "demo_web.yaml", root=ROOT)
    print("preload:", reg.preload(["M1", "M2", "M3", "M4", "M5"]))
    svc = DemoService(reg)

    # JSON tich luy: giu cac cap da co tu cac lan truoc, chi ghi de cac cap
    # duoc cham lai trong lan nay (truoc day moi lan chay deu ghi de het).
    payload: dict = (json.loads(OUT_JSON.read_text(encoding="utf-8"))
                     if OUT_JSON.exists() else {})
    for pid in args.pairs:
        rec = pairs[pid]
        code_a, code_b = codes[rec["a"]], codes[rec["b"]]
        font, max_lines = fig_layout(code_a, code_b)
        analyzed = svc.analyze(code_a, code_b)
        scores = {m["model_id"]: m for m in analyzed["models"]}
        entry = {"a": rec["a"], "b": rec["b"], "label": rec["label"],
                 "note": rec["note"], "scores": {}, "evidence": {}}
        for m in analyzed["models"]:
            entry["scores"][m["model_id"]] = {
                "score": m["score"], "threshold": m["threshold"],
                "verdict": m["verdict"],
            }
        print(f"[{pid}] " + ", ".join(
            f"{mid}={v['score']}" for mid, v in entry["scores"].items()))

        tex: list[str] = [
            f"% ==== cap {pid}: {rec['a']} vs {rec['b']} "
            f"(GT={'clone' if rec['label'] else 'non-clone'}) ====",
        ]
        for mid in MODEL_ORDER:
            sc = scores[mid]
            ev = svc.evidence_for(mid, code_a, code_b).get("evidence") or {}
            entry["evidence"][mid] = {
                "kind": ev.get("kind"), "note": ev.get("note"),
                "n_spans_a": len(ev.get("spans_a", [])),
                "n_spans_b": len(ev.get("spans_b", [])),
                "tok_a": ev.get("tok_a", []), "tok_b": ev.get("tok_b", []),
                "tok_pairs": ev.get("tok_pairs", []),
            }
            short_a = Path(rec["a"]).stem.replace("_", "\\_")
            short_b = Path(rec["b"]).stem.replace("_", "\\_")
            if mid not in render_models:
                continue
            render_model_section(
                tex, mid, MODEL_TITLES[mid], code_a, code_b,
                sc["score"], VERDICT_VI.get(sc["verdict"], sc["verdict"]), ev,
                f"File A ({short_a})", f"File B ({short_b})", font, max_lines)
        out_tex = ROOT / "workspace" / f"demo_suite_{pid}.tex"
        out_tex.write_text("\n".join(tex), encoding="utf-8")
        print(f"  da ghi {out_tex}")

        # MOSS rieng 1 file (giong demo_moss_61.tex).
        out_moss = ROOT / "workspace" / f"demo_suite_moss_{pid}.tex"
        if args.reuse_moss:
            prev = old.get(pid) or {}
            if "B1" not in (prev.get("scores") or {}):
                raise SystemExit(
                    f"--reuse-moss: {OUT_JSON} khong co khoi B1 cua {pid}")
            if not out_moss.exists():
                raise SystemExit(f"--reuse-moss: thieu {out_moss} de giu lai")
            entry["scores"]["B1"] = prev["scores"]["B1"]
            entry["evidence"]["B1"] = prev["evidence"]["B1"]
            print(f"  giu nguyen {out_moss.name} + khoi B1 da co "
                  f"(khong cham lai MOSS)")
        else:
            b1 = scores["B1"]
            bev = svc.evidence_for("B1", code_a, code_b).get("evidence") or {}
            entry["evidence"]["B1"] = {
                "kind": bev.get("kind"), "note": bev.get("note"),
                "report_url": bev.get("report_url"),
                "tok_a": bev.get("tok_a", []), "tok_b": bev.get("tok_b", []),
                "tok_pairs": bev.get("tok_pairs", []),
            }
            from scripts.scan_demo_zip import _latex_minipage

            moss_tex: list[str] = [
                f"% ===== MOSS server cap {pid}: score {fmt_score(b1['score'])}",
                f"\\subsubsection*{{MOSS (server Stanford) — {fmt_score(b1['score'])} "
                f"({VERDICT_VI.get(b1['verdict'], b1['verdict'])})}}",
                f"% GT={'clone' if rec['label'] else 'non-clone'}; "
                f"report={bev.get('report_url')}",
                f"% kind={bev.get('kind')}, note={bev.get('note')}",
            ]
            moss_tex += _latex_minipage(code_a, set(bev.get("tok_a", [])), "File A",
                                        font=font, max_lines=max_lines)
            moss_tex.append("\\hfill")
            moss_tex += _latex_minipage(code_b, set(bev.get("tok_b", [])), "File B",
                                        font=font, max_lines=max_lines)
            out_moss.write_text("\n".join(moss_tex), encoding="utf-8")
            print(f"  da ghi {out_moss}")
        payload[pid] = entry

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"da ghi {OUT_JSON}")


if __name__ == "__main__":
    main()
