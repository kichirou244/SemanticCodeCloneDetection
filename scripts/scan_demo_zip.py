"""Quét demo data.zip + evidence M5 cho báo cáo (chạy local, không train).

Chạy từ gốc repo::

    python scripts/scan_demo_zip.py scan        # 45 cặp data.zip, cả 6 model
    python scripts/scan_demo_zip.py evidence    # evidence M5 (+điểm mọi model) 2 cặp báo cáo

Ra `workspace/demo_scan_all.json` và `workspace/demo_evidence_m5.json`.
Tốn vài chục phút (Joern + 6 model, M5 128M tham số) — để máy chạy nền.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ZIP_PATH = ROOT / "src" / "demo" / "data" / "data.zip"
OUT_SCAN = ROOT / "workspace" / "demo_scan_all.json"
OUT_EVIDENCE = ROOT / "workspace" / "demo_evidence_m5.json"

#: Hai cặp minh họa trong báo cáo: cặp giống hệt trong data.zip + cặp ngược
#: hai bài khác nhau (sample_id trong data/processed/poj104/samples.jsonl).
PAIR_CLONE_FILES = (
    "spit0035_24T1020386_355523.cpp",
    "spit0035_24T1020386_355575.cpp",
)
PAIR_NEG_SAMPLES = ("poj104-train-14763", "poj104-train-16014")
#: Cặp chéo Type-3/4 trong data.zip (dòng tóm tắt "Vệt vàng" ở báo cáo).
PAIR_CROSS_FILES = (
    "spit0035_24T1020386_355523.cpp",
    "spit0035_24T1020331_355172.cpp",
)
OUT_LATEX = ROOT / "workspace" / "demo_m5_latex.txt"


def _service():
    from src.demo.registry import ModelRegistry
    from src.demo.service import DemoService

    reg = ModelRegistry.from_yaml(ROOT / "configs" / "demo_web.yaml", root=ROOT)
    return DemoService(reg)


def cmd_scan() -> None:
    svc = _service()
    data = ZIP_PATH.read_bytes()
    up = svc.upload_zip(data, "data.zip")
    print(f"upload: {len(up['files'])} file, skipped={up['skipped']}")
    result = svc.scan_upload(up["upload_id"])
    OUT_SCAN.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"pairs={result['n_pairs']} kept={result['n_kept']} "
          f"duration_ms={result['duration_ms']}")
    print(f"đã ghi {OUT_SCAN}")
    for row in result["pairs"][:15]:
        cells = {m: c["score"] for m, c in row["scores"].items()}
        print(f"{row['a']} vs {row['b']}: {cells}")


def _zip_codes(*names: str) -> dict[str, str]:
    import zipfile

    with zipfile.ZipFile(ZIP_PATH) as zf:
        # File trong data.zip xuống dòng CRLF: chuẩn hóa về LF ngay khi đọc
        # để offset spans evidence, map token và latex hiển thị khớp nhau.
        return {Path(n).name: zf.read(n).decode("utf-8").replace("\r\n", "\n")
                for n in zf.namelist()}


def _codes_of_pair(svc, name: str) -> tuple[str, str, str, str]:
    if name in ("clone", "cross"):
        blobs = _zip_codes()
        a, b = PAIR_CLONE_FILES if name == "clone" else PAIR_CROSS_FILES
        return a, blobs[a], b, blobs[b]
    from src.demo.samples import CodeIndex

    index = CodeIndex(ROOT / "data" / "processed" / "poj104" / "samples.jsonl")
    a, b = PAIR_NEG_SAMPLES
    norm = lambda s: s.replace("\r\n", "\n")
    return a, norm(index.get(a)), b, norm(index.get(b))


def cmd_evidence() -> None:
    svc = _service()
    payload: dict = {}
    for name in ("clone", "neg", "cross"):
        ida, code_a, idb, code_b = _codes_of_pair(svc, name)
        analyzed = svc.analyze(code_a, code_b)
        entry = {
            "id_a": ida, "id_b": idb,
            "scores": {m["model_id"]: {
                "score": m["score"], "threshold": m["threshold"],
                "verdict": m["verdict"], "label": m["label"],
            } for m in analyzed["models"]},
            "evidence": {},
        }
        for mid in ("B1", "M1", "M2", "M3", "M4", "M5"):
            try:
                ev = svc.evidence_for(mid, code_a, code_b)
                e = ev.get("evidence") or {}
                entry["evidence"][mid] = {
                    "kind": e.get("kind"),
                    "note": e.get("note"),
                    "n_spans_a": len(e.get("spans_a", [])),
                    "n_spans_b": len(e.get("spans_b", [])),
                    "spans_a": e.get("spans_a", []),
                    "spans_b": e.get("spans_b", []),
                    "tok_a": e.get("tok_a", []),
                    "tok_b": e.get("tok_b", []),
                    "tok_pairs": e.get("tok_pairs", []),
                    "error": ev.get("error"),
                }
            except Exception as exc:  # noqa: BLE001 — ghi nhận, không sập script
                entry["evidence"][mid] = {"error": f"{type(exc).__name__}: {exc}"}
        payload[name] = entry
        print(f"[{name}] " + ", ".join(
            f"{m}={v['score']}" for m, v in entry["scores"].items()))
    OUT_EVIDENCE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"đã ghi {OUT_EVIDENCE}")


_LATEX_ESCAPES = [
    ("\\", "\\textbackslash{}"),
    ("{", "\\{"), ("}", "\\}"),
    ("#", "\\#"), ("%", "\\%"), ("&", "\\&"), ("$", "\\$"),
    ("_", "\\_"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ("<", "\\textless{}"), (">", "\\textgreater{}"),
]


def _latex_escape(text: str) -> str:
    # Giữ chỗ backslash trước: thay thẳng sẽ bị các luật \{ \} sau xé
    # "\textbackslash{}" thành "\textbackslash\{\}".
    text = text.replace("\\", "\x00")
    for old, new in _LATEX_ESCAPES[1:]:
        text = text.replace(old, new)
    return text.replace("\x00", "\\textbackslash{}")


def _latex_minipage(
    code: str,
    tok_set: set[int],
    title: str,
    pair_of: dict[int, int] | None = None,
) -> list[str]:
    """Một phía code thành minipage LaTeX: token trong cặp tương ứng →
    \\pline{màu cặp}, token evidence lẻ → \\eline (vàng), còn lại → \\mline.
    Gộp token liên tiếp cùng trạng thái thành một đoạn (khoảng trắng giữa hai
    token tính vào đoạn trước)."""
    from src.demo.alignment.lexer import lex_code

    lines = lex_code(code)
    src_lines = code.split("\n")
    pair_of = pair_of or {}
    out = [
        "\\noindent",
        "\\begin{minipage}[t]{0,49\\textwidth}",
        f"\\footnotesize\\textbf{{{title}}}\\par",
        "{\\footnotesize\\ttfamily",
    ]
    flat = 0
    for li, tokens in enumerate(lines):
        # File trong data.zip xuống dòng \r\n: cắt \r cuối dòng, nếu không
        # \r lọt vào đoạn LaTeX và xé mỗi dòng thành hai.
        text = src_lines[li].removesuffix("\r") if li < len(src_lines) else ""
        segments: list[tuple[tuple, str]] = []
        cur_state: tuple | None = None
        cur_start = 0
        for tok in tokens:
            if flat in pair_of:
                state = ("pair", pair_of[flat] % 8)
            elif flat in tok_set:
                state = ("ev", 0)
            else:
                state = ("plain", 0)
            s, e = tok.col, tok.col + len(tok.text)
            if cur_state is None:
                cur_state, cur_start = state, 0
            elif state != cur_state:
                segments.append((cur_state, text[cur_start:s]))
                cur_state, cur_start = state, s
            flat += 1
        if cur_state is not None:
            segments.append((cur_state, text[cur_start:]))
        parts = "".join(
            _latex_seg(state, seg) for state, seg in segments if seg
        )
        out.append(f"{parts}\\\\")
    out += ["}", "\\end{minipage}"]
    return out


def _latex_seg(state: tuple, seg: str) -> str:
    kind, idx = state
    esc = _latex_escape(seg)
    if kind == "pair":
        return f"\\pline{{{idx}}}{{{esc}}}"
    if kind == "ev":
        return f"\\eline{{{esc}}}"
    return f"\\mline{{{esc}}}"


def _fmt_score(score: float | None) -> str:
    return "---" if score is None else f"{score:.3f}".replace(".", ",")


def cmd_latex() -> None:
    """Sinh khối LaTeX highlight M5 cho 2 cặp báo cáo (không cần GPU/Joern —
    đọc evidence JSON đã có). Ghi ra workspace/demo_m5_latex.txt."""
    payload = json.loads(OUT_EVIDENCE.read_text(encoding="utf-8"))
    titles = {
        "clone": ("File A (355523)", "File B (355575)"),
        "neg": ("File A (bài 30 · 14763)", "File B (bài 33 · 16014)"),
    }
    out: list[str] = []
    for name in ("clone", "neg"):
        entry = payload[name]
        m5 = entry["scores"]["M5"]
        verdict = "clone" if m5["verdict"] == "clone" else "không clone"
        out.append(f"% ===== M5 cặp {name}: {entry['id_a']} vs {entry['id_b']} =====")
        out.append(f"\\subsubsection*{{M5 · Hybrid — {_fmt_score(m5['score'])} ({verdict})}}")
        _, code_a, _, code_b = _codes_of_pair(None, name)
        ev = entry["evidence"]["M5"]
        title_a, title_b = titles[name]
        out += _latex_minipage(code_a, set(ev.get("tok_a", [])), title_a)
        out.append("\\hfill")
        out += _latex_minipage(code_b, set(ev.get("tok_b", [])), title_b)
        out.append("")
        n_a = len(ev.get("spans_a", []))
        n_b = len(ev.get("spans_b", []))
        out.append(f"% M5 {name}: saliency, A {n_a} / B {n_b} chỗ; "
                   f"điểm {m5['score']}, ngưỡng {m5['threshold']}")
        out.append("")
    if "cross" in payload:
        out.append("% ===== span count cặp chéo 355523/355172 =====")
        for mid, ev in payload["cross"]["evidence"].items():
            if isinstance(ev, dict) and "spans_a" in ev:
                out.append(f"% {mid}: kind={ev.get('kind')}, "
                           f"A {len(ev['spans_a'])} / B {len(ev['spans_b'])} chỗ")
    OUT_LATEX.write_text("\n".join(out), encoding="utf-8")
    print(f"đã ghi {OUT_LATEX}")


def main() -> None:
    parser = argparse.ArgumentParser(description="demo scan + evidence M5")
    parser.add_argument("mode", choices=["scan", "evidence", "latex"])
    args = parser.parse_args()
    if args.mode == "scan":
        cmd_scan()
    elif args.mode == "evidence":
        cmd_evidence()
    else:
        cmd_latex()


if __name__ == "__main__":
    main()
