"""Pipeline A1 -> A2 -> A3 cho M1/M2 DiffMul.

A1: calibration (temperature / Platt / isotonic) + ngưỡng chọn bằng OOF trên validation.
A2: SWA-lite (trung bình best + last checkpoint), chọn biến thể bằng validation.
A3: ensemble M1 + M2 với trọng số quét trên OOF validation.

Ngưỡng và trọng số luôn chốt trên validation/OOF; test chỉ dùng để đánh giá cuối
kèm bootstrap CI và McNemar (không tune trên test).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.postprocess.averaging import build_swa_checkpoint, load_checkpoint
from src.postprocess.calibration import fit_calibration
from src.postprocess.inference import infer_logits, load_cache, load_pairs, save_cache
from src.postprocess.metrics import bootstrap_f1, compute_metrics, mcnemar

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "m1": {
        "label": "M1 CodeBERT DiffMul",
        "best": "results/exp_m1_diffmul/best_checkpoint.pt",
        "last": "results/exp_m1_diffmul/last_checkpoint.pt",
    },
    "m2": {
        "label": "M2 GraphCodeBERT DiffMul",
        "best": "results/exp_m2_diffmul/best_checkpoint.pt",
        "last": "results/exp_m2_diffmul/last_checkpoint.pt",
    },
}


@dataclass
class Variant:
    key: str
    name: str
    label: str
    checkpoint: dict[str, Any]
    val_logits: np.ndarray
    test_logits: np.ndarray
    calibration: dict[str, Any]
    val_probs: np.ndarray
    test_probs: np.ndarray
    original_threshold: float
    swa_info: dict[str, Any] | None = None

    @property
    def val_oof_f1(self) -> float:
        return float(self.calibration["chosen_oof_f1"])

    @property
    def threshold(self) -> float:
        return float(self.calibration["chosen_threshold"])


def _logits_for(
    cache_dir: Path,
    name: str,
    split: str,
    checkpoint: dict[str, Any],
    pairs: list[Any],
    batch_size: int,
    max_length: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    cache_path = cache_dir / f"{name}_{split}_logits.npy"
    cached = load_cache(cache_path)
    if cached is not None:
        print(f"[cache] {cache_path.name}: {len(cached)} logits", flush=True)
        return cached
    values = infer_logits(
        checkpoint,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        use_amp=use_amp,
        label=f"{name}/{split}",
    )
    save_cache(cache_path, values)
    return values


def _build_variant(
    key: str,
    name: str,
    label: str,
    checkpoint: dict[str, Any],
    val_pairs: list[Any],
    test_pairs: list[Any],
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
    use_amp: bool,
    n_folds: int,
    seed: int,
) -> Variant:
    val_logits = _logits_for(cache_dir, name, "val", checkpoint, val_pairs, batch_size, max_length, device, use_amp)
    test_logits = _logits_for(cache_dir, name, "test", checkpoint, test_pairs, batch_size, max_length, device, use_amp)

    y_val = np.asarray([int(p.label) for p in val_pairs], dtype=int)
    cal = fit_calibration(val_logits, y_val, n_folds=n_folds, seed=seed)
    method = cal["chosen_method"]
    from src.postprocess.calibration import apply_calibration

    val_probs = apply_calibration(val_logits, method, cal["chosen_params"])
    test_probs = apply_calibration(test_logits, method, cal["chosen_params"])
    original_threshold = float(checkpoint.get("best_threshold", 0.5))
    return Variant(
        key=key,
        name=name,
        label=label,
        checkpoint=checkpoint,
        val_logits=val_logits,
        test_logits=test_logits,
        calibration=cal,
        val_probs=val_probs,
        test_probs=test_probs,
        original_threshold=original_threshold,
        swa_info=checkpoint.get("swa_info"),
    )


def _search_ensemble(
    y_val: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_folds: int,
    seed: int,
    step: float = 0.05,
) -> dict[str, Any]:
    """Quét trọng số w cho A; B nhận 1-w. Chọn w theo OOF F1 trên validation."""
    from src.postprocess.calibration import select_threshold_oof

    best: dict[str, Any] | None = None
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        w = round(float(w), 4)
        mixed = w * probs_a + (1.0 - w) * probs_b
        oof = select_threshold_oof(y_val, mixed, n_folds=n_folds, seed=seed)
        row = {"w_a": w, "w_b": round(1.0 - w, 4), "oof_f1": oof.oof_f1, "oof": oof}
        if best is None or row["oof_f1"] > best["oof_f1"]:
            best = row
    assert best is not None
    return best


def _fmt_variant(v: Variant, y_test: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    from src.postprocess.calibration import sigmoid

    # "raw" = sigmoid(logits) at ngưỡng gốc của checkpoint, để đối chiếu với
    # test_metrics.json cũ. Không được so logits trực tiếp với ngưỡng xác suất.
    raw_metrics = compute_metrics(y_test, sigmoid(v.test_logits), v.original_threshold)
    cal_metrics = compute_metrics(y_test, v.test_probs, v.threshold)
    return {
        "name": v.name,
        "label": v.label,
        "swa": v.swa_info is not None,
        "calibration_method": v.calibration["chosen_method"],
        "calibration_params": v.calibration["chosen_params"],
        "validation": {
            "oof_f1": round(v.val_oof_f1, 4),
            "threshold": v.threshold,
            "nll": v.calibration["fits"][v.calibration["chosen_method"]]["nll"],
            "ece": v.calibration["fits"][v.calibration["chosen_method"]]["ece"],
            "fold_thresholds": v.calibration["fits"][v.calibration["chosen_method"]]["fold_thresholds"],
        },
        "test": {
            "raw_at_original_threshold": raw_metrics,
            "calibrated": cal_metrics,
            "f1_bootstrap": bootstrap_f1(y_test, v.test_probs, v.threshold, n_boot=n_boot, seed=seed),
        },
        "swa_info": v.swa_info,
    }


def run_pipeline(args: Any) -> dict[str, Any]:
    """Chạy toàn bộ A1 -> A2 -> A3, trả về report dict và ghi file."""
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    val_pairs = load_pairs(args.val_file, limit=args.limit_val)
    test_pairs = load_pairs(args.test_file, limit=args.limit_test)
    y_val = np.asarray([int(p.label) for p in val_pairs], dtype=int)
    y_test = np.asarray([int(p.label) for p in test_pairs], dtype=int)
    print(f"Val={len(val_pairs)} pairs | Test={len(test_pairs)} pairs", flush=True)

    model_specs = {key: dict(spec) for key, spec in MODEL_SPECS.items()}
    for key, attr in (("m1", "m1_best"), ("m1", "m1_last"), ("m2", "m2_best"), ("m2", "m2_last")):
        override = getattr(args, attr, None)
        if override:
            model_specs[key]["best" if attr.endswith("best") else "last"] = str(override)

    variants: dict[str, dict[str, Variant]] = {}
    for key, spec in model_specs.items():
        best_path = Path(spec["best"])
        last_path = Path(spec["last"])
        if not best_path.exists():
            raise FileNotFoundError(f"Thiếu checkpoint của {key}: {best_path}")
        if not last_path.exists():
            print(f"[A2] {key}: không có last_checkpoint -> bỏ qua SWA.", flush=True)

        best_ckpt = load_checkpoint(best_path)
        best_variant = _build_variant(
            key,
            f"{key}_best",
            spec["label"],
            best_ckpt,
            val_pairs,
            test_pairs,
            cache_dir,
            args.batch_size,
            args.max_length,
            device,
            args.precision == "fp16",
            args.n_folds,
            args.seed,
        )

        swa_ckpt = (
            build_swa_checkpoint(best_path, last_path, out_dir / f"{key}_swa_checkpoint.pt", alpha=args.swa_alpha)
            if last_path.exists()
            else None
        )
        swa_variant = None
        if swa_ckpt is None:
            print(f"[A2] {key}: best và last cùng epoch -> bỏ qua SWA (weights trùng nhau).", flush=True)
        else:
            swa_variant = _build_variant(
                key,
                f"{key}_swa",
                f"{spec['label']} (SWA best+last)",
                swa_ckpt,
                val_pairs,
                test_pairs,
                cache_dir,
                args.batch_size,
                args.max_length,
                device,
                args.precision == "fp16",
                args.n_folds,
                args.seed,
            )

        variants[key] = {"best": best_variant, "swa": swa_variant}

    # A2: chọn biến thể có OOF F1 validation tốt hơn (ngưỡng chênh tối thiểu 5e-4)
    chosen: dict[str, Variant] = {}
    for key, pair in variants.items():
        best_v = pair["best"]
        swa_v = pair["swa"]
        if swa_v is not None and swa_v.val_oof_f1 > best_v.val_oof_f1 + 5e-4:
            chosen[key] = swa_v
            print(f"[A2] {key}: chọn SWA (val OOF F1 {best_v.val_oof_f1:.4f} -> {swa_v.val_oof_f1:.4f})", flush=True)
        else:
            chosen[key] = best_v
            reason = "không có SWA" if swa_v is None else f"SWA không cải thiện ({swa_v.val_oof_f1:.4f} <= {best_v.val_oof_f1:.4f})"
            print(f"[A2] {key}: giữ best ({reason})", flush=True)

    # A3: ensemble
    ens = _search_ensemble(y_val, chosen["m1"].val_probs, chosen["m2"].val_probs, args.n_folds, args.seed, step=args.weight_step)
    w_a = float(ens["w_a"])
    val_mixed = w_a * chosen["m1"].val_probs + (1.0 - w_a) * chosen["m2"].val_probs
    test_mixed = w_a * chosen["m1"].test_probs + (1.0 - w_a) * chosen["m2"].test_probs
    from src.postprocess.calibration import select_threshold_oof

    ens_oof = select_threshold_oof(y_val, val_mixed, n_folds=args.n_folds, seed=args.seed)
    ens_threshold = float(ens_oof.threshold)
    ens_metrics = compute_metrics(y_test, test_mixed, ens_threshold)

    # Baseline = single model tốt nhất theo val OOF F1
    baseline_key = max(chosen, key=lambda k: chosen[k].val_oof_f1)
    baseline = chosen[baseline_key]
    baseline_metrics = compute_metrics(y_test, baseline.test_probs, baseline.threshold)

    report: dict[str, Any] = {
        "protocol": {
            "calibration_fit": "validation split (bcb_validation.parquet, 3000 pairs)",
            "threshold": "5-fold OOF trên validation, chốt ngưỡng quét F1 trên toàn validation",
            "ensemble_weight": "quét trên OOF validation",
            "test_use": "chỉ đánh giá cuối; Không tune ngưỡng/trọng số trên test",
            "seed": args.seed,
            "n_folds": args.n_folds,
            "precision": args.precision,
        },
        "data": {"n_val": len(val_pairs), "n_test": len(test_pairs)},
        "variants": {},
        "chosen": {
            key: v.name for key, v in chosen.items()
        },
        "ensemble": {
            "weights": {"m1": w_a, "m2": round(1.0 - w_a, 4)},
            "validation": {
                "oof_f1": ens_oof.oof_f1,
                "oof_metrics": ens_oof.oof_metrics,
                "threshold": ens_threshold,
                "weight_grid_best": {
                    "w_a": ens["w_a"],
                    "w_b": ens["w_b"],
                    "oof_f1": ens["oof_f1"],
                },
            },
            "test": ens_metrics,
            "test_f1_bootstrap": bootstrap_f1(y_test, test_mixed, ens_threshold, n_boot=args.n_boot, seed=args.seed),
        },
        "baseline": {
            "model": baseline.name,
            "test": baseline_metrics,
            "test_f1_bootstrap": bootstrap_f1(y_test, baseline.test_probs, baseline.threshold, n_boot=args.n_boot, seed=args.seed),
        },
    }

    for key, pair in variants.items():
        report["variants"][key] = {}
        for name, v in pair.items():
            if v is None:
                continue
            report["variants"][key][name] = _fmt_variant(v, y_test, args.n_boot, args.seed)
            # Lưu chi tiết từng phương pháp calibration (để báo cáo A1)
            report["variants"][key][name]["calibration_methods"] = v.calibration["fits"]

    report["significance"] = {
        "ensemble_vs_baseline": mcnemar(
            y_test,
            baseline.test_probs,
            baseline.threshold,
            test_mixed,
            ens_threshold,
        )
    }

    (out_dir / "postprocess_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(out_dir / "ensemble_test_probs.npy", test_mixed)
    np.save(out_dir / "baseline_test_probs.npy", baseline.test_probs)

    markdown = _render_markdown(report)
    (out_dir / "postprocess_report.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown, flush=True)
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# A1 → A2 → A3: hậu xử lý M1/M2 DiffMul\n")
    lines.append(
        f"- Validation: {report['data']['n_val']:,} cặp | Test: {report['data']['n_test']:,} cặp\n"
        f"- Protocol: {report['protocol']['calibration_fit']}; {report['protocol']['threshold']}\n"
        f"- {report['protocol']['test_use']}\n"
    )

    lines.append("## A1 — Calibration & ngưỡng OOF\n")
    lines.append("| Biến thể | Method | NLL | ECE | Thresh (val) | Thresh OOF | Val OOF F1 | Test F1 (cal) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, pair in report["variants"].items():
        for name, info in pair.items():
            cal = info["calibration_methods"]
            chosen = info["calibration_method"]
            c = cal[chosen]
            tm = info["test"]["calibrated"]
            lines.append(
                f"| {name} | {chosen} | {c['nll']:.4f} | {c['ece']:.4f} | {c['threshold']:.2f} | "
                f"- | {c['oof_f1']:.4f} | {tm['f1']:.4f} |"
            )
    lines.append("")
    lines.append("So sánh các phương pháp calibration (trên cùng validation):\n")
    lines.append("| Biến thể | Method | NLL | ECE | OOF F1 | Ngưỡng |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
    for key, pair in report["variants"].items():
        for name, info in pair.items():
            for method, c in info["calibration_methods"].items():
                lines.append(
                    f"| {name} | {method} | {c['nll']:.4f} | {c['ece']:.4f} | {c['oof_f1']:.4f} | {c['threshold']:.2f} |"
                )
    lines.append("")

    lines.append("## A2 — SWA (best + last)\n")
    for key, pair in report["variants"].items():
        if "swa" in pair:
            b = pair["best"]["validation"]["oof_f1"]
            s = pair["swa"]["validation"]["oof_f1"]
            lines.append(f"- **{key}**: best OOF F1={b:.4f} → SWA OOF F1={s:.4f} → chọn `{report['chosen'][key]}`")
        else:
            lines.append(f"- **{key}**: best và last cùng epoch, bỏ qua SWA → chọn `{report['chosen'][key]}`")
    lines.append("")

    b = report["baseline"]["test"]
    e = report["ensemble"]["test"]
    eci = report["ensemble"]["test_f1_bootstrap"]
    bci = report["baseline"]["test_f1_bootstrap"]
    sig = report["significance"]["ensemble_vs_baseline"]
    lines.append("## A3 — Ensemble M1 + M2\n")
    lines.append(
        f"- Trọng số validation: M1={report['ensemble']['weights']['m1']:.2f}, "
        f"M2={report['ensemble']['weights']['m2']:.2f} | ngưỡng={report['ensemble']['validation']['threshold']:.2f}\n"
        f"- Baseline ({report['baseline']['model']}): F1={b['f1']:.4f} "
        f"(CI95 {bci['ci95'][0]:.4f}–{bci['ci95'][1]:.4f}), P={b['precision']:.4f}, R={b['recall']:.4f}, "
        f"FP={b['fp']}, FN={b['fn']}\n"
        f"- **Ensemble**: F1={e['f1']:.4f} "
        f"(CI95 {eci['ci95'][0]:.4f}–{eci['ci95'][1]:.4f}), P={e['precision']:.4f}, R={e['recall']:.4f}, "
        f"FP={e['fp']}, FN={e['fn']}, ROC-AUC={e['roc_auc']:.4f}\n"
        f"- McNemar (baseline vs ensemble): p={sig['p_value']:.4g}, "
        f"discordant={sig['discordant']}, significant={sig['significant']}\n"
    )
    return "\n".join(lines)
