"""Pipeline cuối (EXP-FINAL-001): đánh giá → chọn phương pháp → suy luận.

Một CLI cho cả ba việc, không train, không đụng checkpoint:

    # 1) Đánh giá & chọn: đọc artifact, xếp hạng theo val F1, ghi
    #    results/final_pipeline/{leaderboard.json,leaderboard.md,selection.json}
    python scripts/final_pipeline.py select

    # 2) Suy luận bằng phương pháp đã chọn (mặc định) trên một cặp file
    python scripts/final_pipeline.py detect --a a.cpp --b b.cpp

    # 3) Ép dùng một model khác trong config (ví dụ so thử với M2-512)
    python scripts/final_pipeline.py detect --a a.cpp --b b.cpp --model M2_512 --json

`select` không cần torch/GPU/Joern. `detect` nạp model thật nên cần checkpoint
(và Joern nếu phương pháp là M4/M5) — xem `docs/evaluation_protocol.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.leaderboard import DEFAULT_CONFIG, run_selection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _resolve_model(cfg: dict, selection: dict, override: str | None) -> str:
    """Chọn khoá model cho `detect`: `--model` > `inference.default_model`."""
    if override:
        return override
    default = cfg.get("inference", {}).get("default_model", "winner")
    if default != "winner":
        return default
    winner = selection.get("winner") or {}
    model = winner.get("model") or ""
    if not model:
        raise SystemExit(
            "Phương pháp thắng không gắn model chạy được (thiếu khoá `model:` "
            f"trong run {winner.get('id')!r}). Thêm vào config hoặc dùng --model."
        )
    return model


def _cmd_select(args: argparse.Namespace) -> int:
    from src.common.config import load_config

    selection = run_selection(args.config, root=ROOT, out_dir=args.out_dir)
    md = (Path(args.out_dir or (ROOT / "results/final_pipeline")) / "leaderboard.md")
    print(md.read_text(encoding="utf-8"))

    winner = selection.get("winner")
    if winner:
        print(f"\n>> Phương pháp chọn: {winner['id']} — {winner['label']}")
        print(f"   val F1 = {winner['metrics'].get('validation', {}).get('f1'):.4f} "
              f"| test F1 = {winner['metrics'].get('test', {}).get('f1'):.4f} "
              f"| ngưỡng = {winner['threshold']}")
    comp = selection.get("comparison")
    if comp:
        p = comp["mcnemar"]["p_value"]
        p_str = "<1e-300" if p == 0.0 else f"{p:.3g}"
        label = {"yes": "khác biệt có ý nghĩa",
                 "no": "CHƯA đủ ý nghĩa",
                 "mixed": "hai kiểm định KHÔNG nhất quán"}[comp.get("significance", "no")]
        print(f"   McNemar vs {comp['b']} trên {comp['split']} "
              f"({comp['n']} cặp): p = {p_str} → {label}")
        boot = comp.get("bootstrap")
        if boot:
            lo, hi = boot["delta_f1_ci95"]
            print(f"   ΔF1 (bootstrap ghép) = {boot['delta_f1_point']:+.4f} "
                  f"CI95 [{lo:+.4f}, {hi:+.4f}]")
    for w in selection.get("warnings", []):
        print(f"   [cảnh báo] {w}")
    default_model = load_config(args.config)["inference"]["default_model"]
    print(f"\n>> `detect` mặc định dùng: {default_model} "
          "(đổi ở configs/final_pipeline.yaml → inference.default_model)")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    from src.common.config import load_config
    from src.demo.registry import ModelRegistry

    cfg = load_config(args.config)
    selection = run_selection(args.config, root=ROOT, out_dir=args.out_dir)
    model_key = _resolve_model(cfg, selection, args.model)

    code_a = Path(args.a).read_text(encoding=args.encoding)
    code_b = Path(args.b).read_text(encoding=args.encoding)

    registry = ModelRegistry(cfg, ROOT)
    scorer = registry.get(model_key)
    result = scorer.score(code_a, code_b)
    out = {
        "model": model_key,
        "label": scorer.label,
        "ok": result.ok,
        "score": result.score,
        "threshold": result.threshold,
        "verdict": result.verdict,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "detail": result.detail,
        "selected_by_pipeline": (selection.get("winner") or {}).get("id"),
    }
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        if not result.ok:
            print(f"{model_key}: LỖI — {result.error}")
            return 1
        print(f"{model_key} ({scorer.label})")
        print(f"  score     = {result.score:.6f}")
        print(f"  threshold = {result.threshold}")
        print(f"  verdict   = {result.verdict}")
        print(f"  latency   = {result.latency_ms} ms")
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", default=None, help="thư mục ghi artifact select")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("select", help="xếp hạng + chọn phương pháp (không cần GPU)")

    detect = sub.add_parser("detect", help="chấm một cặp file bằng model đã chọn")
    detect.add_argument("--a", required=True, help="đường dẫn file code A")
    detect.add_argument("--b", required=True, help="đường dẫn file code B")
    detect.add_argument("--model", default=None, help="khoá model trong config")
    detect.add_argument("--encoding", default="utf-8")
    detect.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "select":
        return _cmd_select(args)
    return _cmd_detect(args)


if __name__ == "__main__":
    raise SystemExit(main())
