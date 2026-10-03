"""Tính mean ± SD và kiểm định so sánh cho các mô hình đã có predictions.

    python scripts/eval_uncertainty.py
    python scripts/eval_uncertainty.py --n-boot 5000 --ref exp002_codebert_poj104

Vì sao cần script riêng
-----------------------
Các runner hiện tại in **một** con số F1 cho mỗi mô hình. Một con số thì không
lấy được độ lệch chuẩn, nên báo cáo không thể viết ``mean ± SD``. Script này đọc
lại ``predictions/test_scores.jsonl`` (đã lưu sẵn score của từng cặp test) rồi
bootstrap — **không cần train lại, không tốn GPU**.

Điều kiện tiên quyết cho kiểm định ghép
--------------------------------------
Hai mô hình phải được đánh giá trên **đúng tập cặp test giống nhau**, cùng thứ tự.
Script kiểm tra lại điều này và **dừng** nếu lệch, vì ghép sai bộ test thì CI tính
ra vẫn đẹp nhưng vô nghĩa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.uncertainty import (  # noqa: E402
    bootstrap_paired, bootstrap_single, mcnemar,
)

RESULTS = Path("results")
OUT_DIR = Path("results/uncertainty")

DEFAULT_EXPERIMENTS = [
    "exp013_moss_poj104",          # B0 MoSS (Winnowing) — baseline chính
    "exp002_codebert_poj104",      # M1 CodeBERT
    "exp003_graphcodebert_poj104",  # M2 GraphCodeBERT
    "exp004_astnn_poj104",         # M3 ASTNN
]

# EXP-001 (TF-IDF) cố ý KHÔNG đưa vào danh sách mặc định: đó là mô hình bag-of-token
# của chính nhóm, dùng làm gốc so sánh thì bất công với các mô hình học sâu.
# Baseline của báo cáo là MOSS (EXP-013). Muốn xem lại TF-IDF thì truyền tên
# experiment vào `--experiments`.


def load_predictions(exp: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Đọc predictions + threshold của một experiment.

    Trả về (pair_id, y_true, y_score). Ngưỡng lấy từ ``threshold.json`` — đã chọn
    trên validation và ghim lại. Không dò lại ngưỡng ở đây.
    """
    d = RESULTS / exp
    # Baseline (EXP-001) ghi thẳng `predictions_test_scores.jsonl` ở thư mục gốc,
    # còn các runner sau đặt trong `predictions/`. Chấp nhận cả hai để EXP-001 vào
    # được bảng so sánh thay vì bị báo "thiếu file" và âm thầm biến mất.
    cand = [d / "predictions" / "test_scores.jsonl",
            d / "predictions_test_scores.jsonl"]
    pred = next((p for p in cand if p.exists()), None)
    thr_file = d / "threshold.json"
    if pred is None:
        raise SystemExit(f"[uncertainty] thieu {cand[0]} (va {cand[1]})")
    if not thr_file.exists():
        raise SystemExit(f"[uncertainty] thieu {thr_file}")
    threshold = json.loads(thr_file.read_text(encoding="utf-8"))["threshold"]

    ids, ys, ss = [], [], []
    with pred.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ids.append(r["pair_id"])
            ys.append(int(r["label"]))
            ss.append(float(r["score"]))
    return ids, np.asarray(ys, dtype=np.int8), np.asarray(ss, dtype=float), threshold


def check_same_pairs(ref_ids: list[str], ids: list[str], exp: str) -> None:
    """Bắt buộc cùng bộ test, cùng thứ tự — điều kiện của kiểm định ghép."""
    if ref_ids == ids:
        return
    if len(ref_ids) != len(ids):
        raise SystemExit(f"[uncertainty] {exp}: so cap test khac nhau "
                         f"({len(ref_ids)} vs {len(ids)}) — khong the ghép")
    only_ref = set(ref_ids) - set(ids)
    only_new = set(ids) - set(ref_ids)
    raise SystemExit(f"[uncertainty] {exp}: tap cap test KHONG giong ref "
                     f"(chi o ref: {len(only_ref)}, chi o moi: {len(only_new)}); "
                     f"vi du {sorted(only_ref)[:2]} / {sorted(only_new)[:2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    ap.add_argument("--ref", default="exp002_codebert_poj104",
                    help="mô hình dùng làm gốc để so sánh ghép")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    loaded = {}
    for exp in args.experiments:
        try:
            loaded[exp] = load_predictions(exp)
        except SystemExit as exc:
            print(f"[uncertainty] bo qua: {exc}")
    if not loaded:
        raise SystemExit("[uncertainty] khong co experiment nao load duoc")

    ref_exp = args.ref if args.ref in loaded else sorted(loaded)[0]
    ref_ids, ref_y, _, _ = loaded[ref_exp]
    print(f"[uncertainty] goc so sanh: {ref_exp}  |  n_test = {len(ref_ids):,}")
    if ref_exp != args.ref:
        print(f"[uncertainty] CHU Y: khong tim thay {args.ref}, dung {ref_exp}")

    report: dict = {
        "n_test": len(ref_ids),
        "n_boot": args.n_boot, "seed": args.seed, "alpha": args.alpha,
        "reference": ref_exp,
        "method": {
            "single_model": "bootstrap percentile tren tap test, nguong ghim",
            "paired": "bootstrap ghep tren cung mot chi so resample cho ca hai",
            "test": "McNemar exact binomial",
            "scope": ("SD nay do kich thuoc tap test (11.000 cap), KHONG do bien "
                      "thien qua lap train. Muon do bien thien lap train thi phai "
                      "train lai nhieu seed -> mean +- SD khac."),
        },
        "models": {}, "paired_vs_reference": {},
    }

    print(f"\n=== mean ± SD (bootstrap {args.n_boot} lan, nguong ghim) ===")
    print(f"{'experiment':<34}{'F1':>8}{'F1 mean±SD':>20}"
          f"{'CI95 F1':>22}{'P':>7}{'R':>7}")
    for exp, (ids, y, s, thr) in loaded.items():
        check_same_pairs(ref_ids, ids, exp)
        st = bootstrap_single(y, s, thr, n_boot=args.n_boot, seed=args.seed,
                              alpha=args.alpha)
        report["models"][exp] = st
        lo, hi = st["f1"]["sd_ci95"]
        print(f"{exp:<34}{st['point']['f1']:>8.4f}"
              f"{st['f1']['mean']:>11.4f} ± {st['f1']['sd']:.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>22}"
              f"{st['point']['precision']:>7.3f}{st['point']['recall']:>7.3f}")

    ref_y, ref_s, ref_thr = loaded[ref_exp][1], loaded[ref_exp][2], loaded[ref_exp][3]
    print(f"\n=== so sanh ghep voi {ref_exp} (bootstrap ghep + McNemar) ===")
    print(f"{'experiment':<34}{'ΔF1':>9}{'CI95 ΔF1':>24}{'McNemar p':>12}{'KL':>6}")
    for exp, (ids, y, s, thr) in loaded.items():
        if exp == ref_exp:
            continue
        # Mỗi mô hình ở ngưỡng đã ghim của RIÊNG nó (đều chọn trên validation).
        pb = bootstrap_paired(y, s, ref_s, thr, ref_thr, n_boot=args.n_boot,
                              seed=args.seed, alpha=args.alpha)
        mc = mcnemar(y, s, ref_s, thr, ref_thr)
        report["paired_vs_reference"][exp] = {"bootstrap": pb, "mcnemar": mc}
        lo, hi = pb["delta_f1_ci95"]
        flag = "co" if pb["significant"] else "KHONG"
        # p-value có thể dưới cả 1e-300 (binomtest trả về 0.0 khi underflow) —
        # in "0" đọc như nghĩa là p=0, tức là sai. In "<1e-300" thay thế.
        p = mc["p_value"]
        p_str = "<1e-300" if p == 0.0 else f"{p:.4g}"
        print(f"{exp:<34}{pb['delta_f1_point']:>+9.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>24}{p_str:>12}{flag:>6}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "uncertainty_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[uncertainty] da ghi {OUT_DIR / 'uncertainty_report.json'}")
    print("[uncertainty] LUU Y: SD o day do kich thuoc tap test, khong do bien "
          "thien lap train. Muon bao cao mean +- SD theo seed thi phai train "
          "lai nhieu seed (M3 re, M1/M2 dat 7h/seed).")


if __name__ == "__main__":
    main()