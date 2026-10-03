"""Chạy MoSS (Winnowing) trên tập cặp POJ-104 — baseline chính của báo cáo.

    python scripts/eval_moss.py
    python scripts/eval_moss.py --k 25 --w 10        # bỏ qua bước dò tham số

Vì sao cần script này
---------------------
Baseline trước đây là TF-IDF, nhưng TF-IDF là mô hình thô (bag-of-token) của
chính nhóm, nên so với nó thì hơi bất công khi kết luận về mô hình học sâu.
MOSS là thuật toán lâu đời và được chấp nhận rộng rãi, nên đây mới là đối thủ
xứng tầm. Chạy hoàn toàn CPU, vài phút, không cần GPU.

Vì sao phải dò tham số
----------------------
Tham số MOSS gốc `k=50, w=100` chọn cho *văn bản dài*. Hàm trong POJ-104 sau
chuẩn hoá chỉ dài khoảng 420 ký tự, tức ~371 k-gram, nên cửa sổ winnow rộng
100×50 = 5000 ký tự là **rộng hơn cả hàm**. Hậu quả đo được: mỗi hàm chỉ còn
khoảng 6 fingerprint, 95% cặp clone không chung fingerprint nào, và similarity
bằng 0 với mọi cặp — F1 đúng bằng của "dự đoán toàn bộ là dương".

Vì vậy phải dò `k, w`. Dò trên **validation**, đóng băng cấu hình, rồi mới chạy
test đúng một lần. Dò trên test là rò rỉ thông tin và làm F1 trên test phồng lên.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import classification_metrics  # noqa: E402
from src.models.baseline.moss import fingerprints, similarity  # noqa: E402

DATA = Path("data")
OUT = Path("results/exp013_moss_poj104")

# (k, w). w phải nhỏ hơn nhiều so với số k-gram của hàm, nên bỏ các giá trị w
# lớn kiểu tham số gốc — chúng đã được đo là không dùng được trên POJ-104.
GRID = [(k, w) for k in (20, 25, 50) for w in (5, 10, 20, 40)]


def load_samples() -> dict[str, str]:
    out: dict[str, str] = {}
    with (DATA / "processed" / "poj104" / "samples.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["sample_id"]] = r["code"]
    return out


def load_pairs(split: str) -> list[dict]:
    with (DATA / "pairs" / "poj104" / f"{split}.jsonl").open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fingerprint_all(codes: dict[str, str], sids: set[str],
                    k: int, w: int) -> dict[str, list[int]]:
    """Fingerprint mọi hàm được dùng, theo đúng `(k, w)`.

    Gọi một lần cho cả validation lẫn test: cùng `(k, w)` thì dùng chung được,
    và chỉ vài nghìn hàm chứ không phải 22.000 lần gọi lặp.
    """
    out: dict[str, list[int]] = {}
    for i, sid in enumerate(sorted(sids), 1):
        out[sid] = fingerprints(codes[sid], k=k, w=w)
        if i % 2000 == 0:
            print(f"[moss]   fingerprint {i:,}/{len(sids):,}", flush=True)
    return out


def score_pairs(pairs, fps) -> np.ndarray:
    return np.array([similarity(fps[r["code_a_id"]], fps[r["code_b_id"]])
                     for r in pairs], dtype=float)


def best_threshold(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Ngưỡng tối đa hoá F1. Chỉ thử trên các giá trị score đã thấy — ngưỡng chỉ
    có ý nghĩa khi nằm giữa hai score liên tiếp."""
    cands = np.unique(s)
    if cands.size > 512:  # tránh O(n_thresholds × n_pairs) quá lớn
        cands = np.quantile(cands, np.linspace(0, 1, 512))
    best_f1, best_thr = -1.0, 0.5
    for thr in cands:
        pred = s >= thr
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(thr)
    return best_thr, best_f1


def f1_at(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    pred = s >= thr
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="bỏ qua dò, dùng k này")
    ap.add_argument("--w", type=int, default=None, help="bỏ qua dò, dùng w này")
    args = ap.parse_args()

    t0 = time.time()
    codes = load_samples()
    val_pairs, test_pairs = load_pairs("validation"), load_pairs("test")
    sids = {r["code_a_id"] for r in val_pairs} | {r["code_b_id"] for r in val_pairs} \
        | {r["code_a_id"] for r in test_pairs} | {r["code_b_id"] for r in test_pairs}
    vy = np.array([int(r["label"]) for r in val_pairs], dtype=np.int8)
    ty = np.array([int(r["label"]) for r in test_pairs], dtype=np.int8)
    print(f"[moss] {len(codes):,} ham | {len(val_pairs):,} cap val | "
          f"{len(test_pairs):,} cap test | {len(sids):,} ham duoc dung")

    # ---- bước 1: dò (k, w) trên validation
    if args.k is not None and args.w is not None:
        grid = [(args.k, args.w)]
        print("[moss] chiay tham so da duoc chon, bo qua buoc do")
    else:
        grid = GRID
    print(f"\n=== do (k, w) tren VALIDATION ({len(grid)} to hop) ===")
    print(f"{'k':>4}{'w':>5}{'n_fp median':>13}{'sim>0 (clone)':>15}{'F1 val':>9}")
    results = []
    for k, w in grid:
        fps = fingerprint_all(codes, sids, k, w)
        vs = score_pairs(val_pairs, fps)
        thr, f1 = best_threshold(vy, vs)
        pos = vs[vy == 1]
        n_fp = statistics_median([len(fps[r]) for r in sorted(sids)])
        frac = float(np.mean(pos > 0))
        print(f"{k:>4}{w:>5}{n_fp:>13.0f}{frac:>14.1%}{f1:>9.4f}")
        results.append((f1, k, w, thr, n_fp, frac))
        if (k, w) != grid[-1]:
            del fps  # giữ RSS thấp khi dò nhiều tổ hợp

    results.sort(reverse=True)
    _, k, w, thr, n_fp, frac = results[0]
    print(f"\n[moss] chon k={k}, w={w} (F1 validation cao nhat)")

    # ---- bước 2: chạy test đúng một lần với cấu hình đã đóng băng
    fps = fingerprint_all(codes, sids, k, w)
    ts = score_pairs(test_pairs, fps)
    test_f1 = f1_at(ty, ts, thr)
    metrics = {
        "validation": {"f1": results[0][0], "threshold": thr, "n": int(vy.size)},
        "test": classification_metrics(ty, ts, thr),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    with (OUT / "predictions" / "test_scores.jsonl").open("w", encoding="utf-8") as fh:
        for r, score in zip(test_pairs, ts):
            fh.write(json.dumps({"pair_id": r["pair_id"], "label": int(r["label"]),
                                 "score": round(float(score), 6)}) + "\n")
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "threshold.json").write_text(json.dumps(
        {"threshold": thr, "f1": results[0][0], "method": "validation_f1",
         "split": "validation", "k": k, "w": w}, indent=2), encoding="utf-8")
    (OUT / "run_metadata.json").write_text(json.dumps({
        "experiment": "EXP-013", "model": "B0 MoSS (Winnowing)",
        "k": k, "w": w, "grid_searched_on": "validation",
        "median_fingerprints_per_function": n_fp,
        "frac_clone_pairs_with_any_match": frac,
        "similarity": "containment (JPlag-family), chuan hoa theo ben it nguon hon",
        "threshold_source": "validation_f1 (ghim, khong do lai tren test)",
        "runtime_seconds": round(time.time() - t0, 1),
        "device": "cpu", "n_test": int(ty.size),
        "note": ("Tham so goc k=50,w=100 khong dung duoc tren POJ-104: ham ngan, "
                 "cua so winnow rong hon ca ham nen moi ham chi con ~6 fingerprint."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    t = metrics["test"]
    print(f"[moss] test: median n_fp={n_fp:.0f} | {frac:.1%} cap clone co chung fingerprint")
    print(f"[moss] TEST  P={t['precision']:.4f} R={t['recall']:.4f} F1={t['f1']:.4f} "
          f"ROC-AUC={t['roc_auc']:.4f}")
    print(f"[moss] xong trong {time.time() - t0:.1f}s -> {OUT}")


def statistics_median(xs) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2)


if __name__ == "__main__":
    main()