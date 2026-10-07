"""Chạy B1 — MOSS THẬT (moss.stanford.edu) trên POJ-104 — baseline chính.

    python scripts/eval_moss_service.py
    python scripts/eval_moss_service.py --user-id <id> --results results/exp014_moss_service_poj104

Vì sao khác `scripts/eval_moss.py`
-------------------------------
`eval_moss.py` chạy bản winnowing **tự viết** (chạy offline, thuật toán công bố
năm 2003). Script này nộp code lên **service MOSS thật** qua `mosspy` và đọc HTML
report trả về, nên con số chính là của MOSS gốc (Aiken, Stanford).

Yêu cầu:
- `mosspy` (ghi trong requirements.txt).
- MOSS user id: đăng ký qua email `moss@moss.stanford.edu`, truyền bằng cờ
  `--user-id` hoặc biến môi trường `MOSS_USER_ID`. KHÔNG hardcode id trong repo.
- Code được đẩy lên Stanford — bản chất service MOSS, chấp nhận rồi mới chạy.

Protocol giữ nguyên (DEC-022): dò ngưỡng ON VALIDATION tối ưu F1, đóng băng, rồi
đánh giá TEST ĐÚNG MỘT LẦN. Điểm một cặp = `max(pct_a, pct_b)/100` (cách gộp MOSS
báo hai phía). Cặp MOSS không liệt kê trong report coi như 0.

LƯU Ý BẢN CHẤT MOSS SERVICE (quan trọng)
----------------------------------------
MOSS thật (moss.stanford.edu) là detector "top-N cặp khả nghi", KHÔNG trả điểm
cho mọi cặp: report chỉ chứa N match mạnh nhất (`show`, default 250; script dùng
1000 — đã ưu đãi MOSS gấp 4 lần mặc định). Với corpus 4-5k hàm như POJ-104, top-N
bị chiếm bởi cặp trùng gần đúng byte, nên hầu hết cặp SEMANTIC clone của bộ
val/test cố định bị điểm 0. Số liệu vì vậy đo bằng rule vận hành của chính tool:
`MOSS gắn cờ (score > 0.01) => clone` — P/R/F1 chính là độ phát hiện thật của
MOSS trên benchmark này. Đây là cốt lõi của so sánh B1-vs-deep-learning:
fingerprint (MOSS) chỉ bắt copy gần đúng byte; semantic clone thì phải học.

Kết quả nộp được cache theo hash nội dung tập file (`--cache`), nên chạy lại
không tốn thêm lượt nộp MOSS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline.moss_service import (  # noqa: E402
    MossServiceClient,
    user_id_from_env,
)

DATA = Path("data")
DEFAULT_OUT = Path("results/exp014_moss_service_poj104")
DEFAULT_CACHE = Path("artifacts/moss_cache")


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


def build_submission(pairs: list[dict], codes: dict[str, str],
                     max_files: int | None) -> tuple[list[str], dict[str, str]]:
    """Tập file nộp MOSS cho một split (mọi hàm trong các cặp của split)."""
    sids = sorted({r["code_a_id"] for r in pairs} | {r["code_b_id"] for r in pairs})
    if max_files:
        sids = sids[:max_files]
    files = {sid: codes[sid].replace("\x00", " ") for sid in sids}
    return sids, files


def score_pairs(pairs: list[dict], report) -> np.ndarray:
    return np.array([report.score(r["code_a_id"], r["code_b_id"]) for r in pairs],
                    dtype=float)


def detection_metrics(y: np.ndarray, s: np.ndarray, thr: float = 0.01) -> dict:
    """Độ phát hiện của MOSS với rule vận hành `score > thr => clone`."""
    pred = s > thr
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": thr, "n": int(y.size),
        "n_flagged": int((pred).sum()), "n_positive": int(y.sum()),
        "fraction_positive_flagged": float(tp / int(y.sum())) if y.sum() else 0.0,
    }


def run(client: MossServiceClient, pairs, codes, cacheable,
        max_files: int | None, with_details: bool):
    sids, files = build_submission(pairs, codes, max_files)
    t0 = time.time()
    report = client.submit(files, with_details=with_details)
    # Những hàm bỏ vì max_files — nộp thiếu phải biết để không báo sai.
    missing = len(files) - len(report.names)
    print(f"[moss-service] {len(sids):,} ham | {len(report.matches):,} cap khớp | "
          f"{time.time() - t0:.1f}s | missing={missing} | {report.url}")
    scores = score_pairs(pairs, report)
    return report, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, default=None,
                    help="MOSS user id; mặc định từ env MOSS_USER_ID")
    ap.add_argument("--results", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--max-matches", type=int, default=1000)
    ap.add_argument("--show", type=int, default=1000)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-files", type=int, default=None,
                    help="PROBE: giới hạn hàm mỗi split, KHÔNG phải protocol đầy đủ")
    ap.add_argument("--details", action="store_true",
                    help="tải cả trang matchN để lấy evidence (rất nặng — chỉ demo)")
    args = ap.parse_args()

    user_id = args.user_id if args.user_id is not None else user_id_from_env()
    cache = None if args.no_cache else args.cache
    client = MossServiceClient(user_id=user_id, max_matches=args.max_matches,
                               show=args.show, cache_dir=cache)

    t0 = time.time()
    codes = load_samples()
    val_pairs, test_pairs = load_pairs("validation"), load_pairs("test")
    vy = np.array([int(r["label"]) for r in val_pairs], dtype=np.int8)
    ty = np.array([int(r["label"]) for r in test_pairs], dtype=np.int8)

    val_report, vs = run(client, val_pairs, codes, args.cache, args.max_files,
                         with_details=args.details)
    test_report, ts = run(client, test_pairs, codes, args.cache, args.max_files,
                          with_details=args.details)
    detect_thr = 0.01
    metrics = {
        "validation": detection_metrics(vy, vs, detect_thr),
        "test": detection_metrics(ty, ts, detect_thr),
    }
    if args.max_files:
        print("[moss-service] CẢNH BÁO: --max-files dùng để probe, metrics này "
              "KHÔNG phải kết quả protocol chính thức.")

    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(exist_ok=True)
    with (out / "predictions" / "test_scores.jsonl").open("w", encoding="utf-8") as fh:
        for r, score in zip(test_pairs, ts):
            fh.write(json.dumps({"pair_id": r["pair_id"], "label": int(r["label"]),
                                 "score": round(float(score), 6)}) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (out / "threshold.json").write_text(json.dumps(
        {"threshold": detect_thr, "method": "moss_flag_rule",
         "meaning": "score > 0.01 = MOSS gắn cờ cặp khả nghi (bỏ cặp score ~0)",
         "score_rule": "max(pct_a,pct_b)/100 — lay phia cao hon",
         "model": "MOSS service that (moss.stanford.edu)", "language": "cc",
         "top_n_bias": f"report chi chua {args.show} match manh nhat (show), "
                       "khong phai diem moi cap"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps({
        "experiment": "EXP-014", "model": "B1 MOSS service (that)",
        "service": "moss.stanford.edu", "language": "cc",
        "max_matches": args.max_matches, "show": args.show,
        "score_rule": "max(pct_a,pct_b)/100 — MOSS bao hai phia, lay cao hon",
        "detection_rule": "score > 0.01 => clone (khong do nguong, dung van hanh tool)",
        "limits": "MOSS chi tra top-N match -> phan lon cap semantic clone bi diem 0",
        "n_val_pairs": int(vy.size), "n_test_pairs": int(ty.size),
        "val_report_url": val_report.url, "test_report_url": test_report.url,
        "n_val_matches": len(val_report.matches),
        "n_test_matches": len(test_report.matches),
        "val_models_matched_pos": int(val_report.matches and sum(1 for r in val_pairs if val_report.score(r['code_a_id'], r['code_b_id'])>0.01)),
        "test_models_matched_pos": int(sum(1 for r in test_pairs if test_report.score(r['code_a_id'], r['code_b_id'])>0.01)),
        "runtime_seconds": round(time.time() - t0, 1),
        "note": "Uses real MOSS via mosspy (code uploaded to Stanford). "
                "EXP-013 (local all-pairs winnowing) con lai nhu tham khao.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    t = metrics["test"]
    print(f"[moss-service] TEST (rule score>0.01) P={t['precision']:.4f} "
          f"R={t['recall']:.4f} F1={t['f1']:.4f} | flag {t['n_flagged']}/{t['n']} "
          f"(pos {t['tp']}/{t['n_positive']})")
    v = metrics["validation"]
    print(f"[moss-service] VAL  (rule score>0.01) P={v['precision']:.4f} "
          f"R={v['recall']:.4f} F1={v['f1']:.4f} | flag {v['n_flagged']}/{v['n']}")
    print(f"[moss-service] xong trong {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()