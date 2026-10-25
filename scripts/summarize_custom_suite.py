"""Tong hop danh gia bo test `custom_suite` theo trung binh + do lech chuan.

Doc `workspace/custom_suite_scores.json` (do `score_custom_suite.py` ghi) va
`data/custom_suite/pairs.jsonl`, roi xuat:

- `workspace/custom_suite_summary.json` — so lieu day du
- `workspace/custom_suite_summary.md`   — bang Markdown de doc nhanh
- `workspace/custom_suite_summary.tex`  — bang LaTeX de \\input vao bao cao

Hai kieu "mean ± SD" duoc tach ro:

1. **Diem theo cap** (`score mean ± SD`): trung binh cong va do lech chuan cua
   diem tho tren cac cap trong nhom. Day la do phan tan diem, tra loi cau hoi
   "nhom non-clone co de khong".
2. **F1 bootstrap** (`F1 mean ± SD`): resample cac cap (co hoan lai) roi tinh
   lai F1, lay trung binh + do lech chuan cua phan phoi bootstrap. Day la do
   bat dinh do tap test huu han, tai dung `src/evaluation/uncertainty.py`.

Vi mot so mo hinh (B1/MOSS that) chi co diem tren bo loi cu, script in them bang
"common": giao cua cac cap co du diem cho **tat ca** mo hinh duoc chon, de so
sanh cong bang.

Chay::

    python scripts/summarize_custom_suite.py
    python scripts/summarize_custom_suite.py --models B1 M1 M2 M3 M4 M5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.uncertainty import bootstrap_single  # noqa: E402

DATA_DIR = ROOT / "data" / "custom_suite"
DEFAULT_SCORES = ROOT / "workspace" / "custom_suite_scores.json"
DEFAULT_OUT = ROOT / "workspace" / "custom_suite_summary"
MODEL_ORDER = ["B1", "M1", "M2", "M3", "M4", "M5"]
CATEGORY_ORDER = ["clone_de", "clone_kho", "nonclone_kho", "nonclone_de"]
CATEGORY_VI = {
    "clone_de": "Clone dễ",
    "clone_kho": "Clone khó",
    "nonclone_kho": "Không clone khó",
    "nonclone_de": "Không clone dễ",
}


def load_pairs(path: Path = DATA_DIR / "pairs.jsonl") -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["pair_id"]] = r
    return rows


def load_scores(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def threshold_for(model: str, scores: dict[str, dict]) -> float:
    for row in scores.values():
        cell = (row.get("scores") or {}).get(model)
        if cell and cell.get("threshold") is not None:
            return float(cell["threshold"])
    raise SystemExit(f"khong tim thay threshold cua {model}")


def collect(model: str, scores: dict[str, dict]) -> list[dict]:
    """Cac cap co diem hop le cua mot mo hinh."""
    out = []
    for pid, row in scores.items():
        cell = (row.get("scores") or {}).get(model)
        if not cell or not cell.get("ok", False):
            continue
        out.append({
            "pair_id": pid,
            "label": int(row.get("label", 0)),
            "category": row.get("category", "?"),
            "score": float(cell["score"]),
            "threshold": float(cell["threshold"]),
        })
    return out


def _mean_sd(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), sd


def _metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    y = np.asarray([r["label"] for r in rows], dtype=np.int8)
    s = np.asarray([r["score"] for r in rows], dtype=float)
    thr = float(rows[0]["threshold"])
    pred = s >= thr
    tp = int(np.sum((y == 1) & pred))
    fp = int(np.sum((y == 0) & pred))
    tn = int(np.sum((y == 0) & ~pred))
    fn = int(np.sum((y == 1) & ~pred))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    acc = (tp + tn) / len(rows) if rows else 0.0
    cl = [r["score"] for r in rows if r["label"] == 1]
    nc = [r["score"] for r in rows if r["label"] == 0]
    cm, csd = _mean_sd(cl)
    nm, nsd = _mean_sd(nc)
    return {
        "n": len(rows), "n_clone": len(cl), "n_nonclone": len(nc),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "threshold": thr,
        "score_clone_mean": cm, "score_clone_sd": csd,
        "score_nonclone_mean": nm, "score_nonclone_sd": nsd,
        "score_gap": (cm - nm) if not (math.isnan(cm) or math.isnan(nm)) else float("nan"),
    }


def _bootstrap(rows: list[dict], n_boot: int, seed: int) -> dict:
    y = np.asarray([r["label"] for r in rows], dtype=np.int8)
    s = np.asarray([r["score"] for r in rows], dtype=float)
    if len(rows) < 2 or len(set(y.tolist())) < 2:
        return {}
    st = bootstrap_single(y, s, float(rows[0]["threshold"]),
                          n_boot=n_boot, seed=seed)
    return {k: st[k] for k in ("precision", "recall", "f1")}


def summarize_model(model: str, rows: list[dict], n_boot: int, seed: int) -> dict:
    out = {"all": _metrics(rows)}
    out["all"]["bootstrap"] = _bootstrap(rows, n_boot, seed)
    out["by_category"] = {}
    for cat in CATEGORY_ORDER:
        sub = [r for r in rows if r["category"] == cat]
        if sub:
            out["by_category"][cat] = _metrics(sub)
    return out


def fmt(x: float, nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fmt_pm(mean: float, sd: float, nd: int = 3) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def fmt_vn(x: float, nd: int = 3) -> str:
    """So kieu Viet Nam (dau phay) cho LaTeX."""
    return fmt(x, nd).replace(".", ",")


def fmt_pm_vn(mean: float, sd: float, nd: int = 3) -> str:
    return fmt_pm(mean, sd, nd).replace(".", ",")


def table_overall_md(models: list[str], summary: dict, scope: str,
                     title: str) -> list[str]:
    lines = [f"### {title}", "",
             "| Mô hình | n | Acc | P | R | F1 | F1 mean ± SD (bootstrap) | "
             "Điểm clone mean ± SD | Điểm non-clone mean ± SD | Khoảng cách |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for m in models:
        s = summary[m][scope]
        if not s.get("n"):
            lines.append(f"| {m} | 0 | — | — | — | — | — | — | — | — |")
            continue
        boot = s.get("bootstrap") or {}
        f1b = boot.get("f1") or {}
        lines.append(
            f"| **{m}** | {s['n']} | {fmt(s['accuracy'])} | {fmt(s['precision'])} | "
            f"{fmt(s['recall'])} | {fmt(s['f1'])} | "
            f"{fmt_pm(f1b.get('mean', float('nan')), f1b.get('sd', float('nan')))} | "
            f"{fmt_pm(s['score_clone_mean'], s['score_clone_sd'])} | "
            f"{fmt_pm(s['score_nonclone_mean'], s['score_nonclone_sd'])} | "
            f"{fmt(s['score_gap'])} |")
    return lines


def table_category_md(models: list[str], summary: dict, key: str,
                      title: str, nd: int = 3) -> list[str]:
    lines = [f"### {title}", "",
             "| Nhóm | n | " + " | ".join(models) + " |",
             "| :--- | ---: | " + " | ".join("---:" for _ in models) + " |"]
    for cat in CATEGORY_ORDER:
        n = max(summary[m]["by_category"].get(cat, {}).get("n", 0)
                for m in models)
        cells = []
        for m in models:
            s = summary[m]["by_category"].get(cat, {})
            if not s.get("n"):
                cells.append("—")
            elif key == "score":
                use_clone = s["n_clone"] > 0
                cells.append(fmt_pm(s["score_clone_mean"] if use_clone
                                    else s["score_nonclone_mean"],
                                    s["score_clone_sd"] if use_clone
                                    else s["score_nonclone_sd"], nd))
            elif key == "f1":
                cells.append(fmt(s["f1"], nd))
            elif key == "errors":
                cells.append(str(s["fp"] + s["fn"]))
            else:
                cells.append(fmt(s.get(key, float("nan")), nd))
        lines.append(f"| {CATEGORY_VI.get(cat, cat)} | {n} | " + " | ".join(cells) + " |")
    return lines


def table_overall_tex(models: list[str], summary: dict, scope: str,
                      caption: str, label: str, complete: bool) -> list[str]:
    lines = [
        "\\begin{table}[htbp]\\centering\\scriptsize",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Mô hình & n & Acc & P & R & F1 & F1 $\\pm$ SD & "
        "$\\bar{s}_{cl}$ & $\\bar{s}_{nc}$ \\\\",
        "\\midrule",
    ]
    for m in models:
        s = summary[m][scope]
        if not s.get("n"):
            lines.append(f"{m} & 0 & -- & -- & -- & -- & -- & -- & -- \\\\")
            continue
        f1b = (s.get("bootstrap") or {}).get("f1") or {}
        lines.append(
            f"{m} & {s['n']} & {fmt_vn(s['accuracy'])} & {fmt_vn(s['precision'])} & "
            f"{fmt_vn(s['recall'])} & {fmt_vn(s['f1'])} & "
            f"{fmt_pm_vn(f1b.get('mean', float('nan')), f1b.get('sd', float('nan')))} & "
            f"{fmt_pm_vn(s['score_clone_mean'], s['score_clone_sd'])} & "
            f"{fmt_pm_vn(s['score_nonclone_mean'], s['score_nonclone_sd'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def table_category_tex(models: list[str], summary: dict, key: str,
                       caption: str, label: str, nd: int = 3) -> list[str]:
    """Bang gon: moi dong mot mo hinh, moi cot mot nhom cap."""
    lines = [
        "\\begin{table}[htbp]\\centering\\scriptsize",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Mô hình & " + " & ".join(CATEGORY_VI[c] for c in CATEGORY_ORDER) + " \\\\",
        "\\midrule",
        "Số cặp & " + " & ".join(
            str(max(summary[m]["by_category"].get(c, {}).get("n", 0)
                    for m in models)) for c in CATEGORY_ORDER) + " \\\\",
        "\\midrule",
    ]
    for m in models:
        cells = []
        for cat in CATEGORY_ORDER:
            s = summary[m]["by_category"].get(cat, {})
            if not s.get("n"):
                cells.append("--")
            elif key == "score":
                use_clone = s["n_clone"] > 0
                cells.append(fmt_pm_vn(
                    s["score_clone_mean"] if use_clone else s["score_nonclone_mean"],
                    s["score_clone_sd"] if use_clone else s["score_nonclone_sd"], nd))
            elif key == "f1":
                cells.append(fmt_vn(s["f1"], nd))
            elif key == "errors":
                cells.append(str(s["fp"] + s["fn"]))
            else:
                cells.append(fmt_vn(s.get(key, float("nan")), nd))
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="tong hop custom_suite")
    ap.add_argument("--scores", default=str(DEFAULT_SCORES))
    ap.add_argument("--models", nargs="*", default=MODEL_ORDER)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="tien to file output (khong kem duoi)")
    args = ap.parse_args()

    pairs = load_pairs()
    scores = load_scores(Path(args.scores))
    models = [m for m in args.models if any(
        (row.get("scores") or {}).get(m, {}).get("ok") for row in scores.values())]
    if not models:
        raise SystemExit("khong co mo hinh nao co diem")

    collected = {m: collect(m, scores) for m in models}
    summary: dict[str, dict] = {}
    for m in models:
        summary[m] = summarize_model(m, collected[m], args.n_boot, args.seed)
        # Them threshold (de tai lap) va so cap theo nhom.
        summary[m]["threshold"] = threshold_for(m, scores)

    # "common": giao cac cap co du diem cho MOI mo hinh duoc chon.
    id_sets = [set(r["pair_id"] for r in collected[m]) for m in models]
    common_ids = set.intersection(*id_sets) if id_sets else set()
    for m in models:
        rows = [r for r in collected[m] if r["pair_id"] in common_ids]
        summary[m]["common"] = _metrics(rows)
        summary[m]["common"]["bootstrap"] = _bootstrap(rows, args.n_boot, args.seed)

    payload = {
        "source": str(Path(args.scores)),
        "models": models,
        "n_pairs_file": len(pairs),
        "n_nonclone_file": sum(1 for r in pairs.values() if int(r["label"]) == 0),
        "n_clone_file": sum(1 for r in pairs.values() if int(r["label"]) == 1),
        "n_common": len(common_ids),
        "meaning": {
            "score_mean_sd": "trung binh + do lech chuan cua diem tho tren cac cap",
            "f1_bootstrap": "mean + SD cua F1 khi resample cap (do bat dinh tap test)",
        },
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    md: list[str] = [
        "# Tổng hợp bộ test `custom_suite` (mean ± SD)",
        "",
        f"- Nguồn điểm: `{args.scores}`",
        f"- Tổng cặp trong manifest: **{payload['n_pairs_file']}** "
        f"({payload['n_clone_file']} clone + {payload['n_nonclone_file']} non-clone)",
        f"- Cặp có đủ điểm cho mọi mô hình (common): **{len(common_ids)}**",
        "- `Điểm clone/non-clone mean ± SD`: trung bình và độ lệch chuẩn của điểm thô "
        "trên các cặp (độ phân tán điểm, không phải độ bất định của F1).",
        "- `F1 mean ± SD`: bootstrap 2.000 lần trên các cặp, ngưỡng đóng băng.",
        "",
    ]
    md += table_overall_md(models, summary, "all",
                           "Toàn bộ cặp có điểm (mỗi mô hình có thể khác n)")
    md += [""]
    md += table_category_md(models, summary, "score",
                            "Điểm trung bình ± SD theo nhóm (nhóm clone dùng điểm clone, "
                            "nhóm non-clone dùng điểm non-clone)")
    md += [""]
    md += table_category_md(models, summary, "errors",
                            "Số lỗi (FP + FN) theo nhóm")
    md += [""]
    md += table_overall_md(models, summary, "common",
                           f"So sánh công bằng trên {len(common_ids)} cặp chung")
    md += [""]
    (out.with_suffix(".md")).write_text("\n".join(md), encoding="utf-8")

    tex: list[str] = [
        "% Sinh boi scripts/summarize_custom_suite.py — dung \\input{}",
        "% F1 mean±SD la bootstrap tren cap; diem clone/non-clone mean±SD la do phan tan diem.",
        "",
    ]
    tex += table_overall_tex(
        models, summary, "all",
        f"Toàn bộ cặp có điểm ({payload['n_pairs_file']} cặp trong manifest; "
        "mỗi mô hình có thể khác n) --- $\\bar{s}_{cl}/\\bar{s}_{nc}$ là điểm "
        "trung bình clone/non-clone $\\pm$ SD",
        "tab:suite-all", complete=False)
    tex += table_category_tex(models, summary, "score",
                              "Điểm trung bình $\\pm$ SD theo nhóm cặp "
                              "(nhóm clone dùng điểm clone, nhóm non-clone dùng điểm non-clone)",
                              "tab:suite-cat-score")
    tex += table_category_tex(models, summary, "errors",
                              "Số lỗi (FP + FN) theo nhóm",
                              "tab:suite-cat-err")
    tex += table_overall_tex(
        models, summary, "common",
        f"So sánh công bằng trên {len(common_ids)} cặp chung của mọi mô hình --- "
        "$\\bar{s}_{cl}/\\bar{s}_{nc}$ là điểm trung bình clone/non-clone $\\pm$ SD",
        "tab:suite-common", complete=True)
    (out.with_suffix(".tex")).write_text("\n".join(tex), encoding="utf-8")

    print(f"da ghi {json_path}")
    print(f"da ghi {out.with_suffix('.md')}")
    print(f"da ghi {out.with_suffix('.tex')}")
    print(f"common = {len(common_ids)} cap; models = {models}")
    for m in models:
        s = summary[m]["all"]
        if s.get("n"):
            print(f"  {m}: n={s['n']:3d} F1={s['f1']:.4f} "
                  f"score clone={fmt_pm(s['score_clone_mean'], s['score_clone_sd'])} "
                  f"non-clone={fmt_pm(s['score_nonclone_mean'], s['score_nonclone_sd'])}")


if __name__ == "__main__":
    main()
