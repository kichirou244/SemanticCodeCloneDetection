"""EXP-002 — M1 CodeBERT fine-tuning trên POJ-104.

Chạy (trên máy GPU của user):
  preflight: python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml --mode preflight --device cuda
  smoke:     ... --mode smoke --device cpu
  train:     ... --mode train --device cuda
  evaluate:  ... --mode evaluate --device cuda

Protocol (§35): train chỉ update trên train; val chỉ eval; test untouched tới
final evaluation; threshold trên val (F1); best checkpoint theo val F1.
"""

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.data.access import stream_codes, stream_jsonl
from src.experiments.exp001_tfidf import (
    select_threshold,
    sha256_file,
    split_metrics,
    validate_dataset,
    run_metadata,
)
from src.models.codebert_pair import (
    CodeBERTPairClassifier,
    collate_pair_batch,
    load_tokenizer,
)

SMOKE_TRAIN_PAIRS = 256
SMOKE_VAL_PAIRS = 128


# ---------------------------------------------------------------- data
def load_code_map(processed: Path, splits: list[str]) -> dict[str, str]:
    codes: dict[str, str] = {}
    for row in stream_jsonl(processed):
        if row["split"] in splits:
            codes[row["sample_id"]] = row["code"]
    return codes


def load_pair_refs(pairs_path: Path) -> list[dict]:
    return [
        {
            "pair_id": r["pair_id"],
            "a": r["code_a_id"],
            "b": r["code_b_id"],
            "problem_a": r.get("problem_a", ""),
            "problem_b": r.get("problem_b", ""),
            "label": r["label"],
        }
        for r in stream_jsonl(pairs_path)
    ]


def truncation_analysis(tokenizer, codes: dict[str, str], max_length: int) -> dict:
    """Đếm codes vượt max_length (tokenize không truncate, theo batch)."""
    import warnings

    ids = list(codes)
    total_trunc, total = 0, len(ids)
    for i in range(0, len(ids), 512):
        batch = [codes[k] for k in ids[i : i + 512]]
        with warnings.catch_warnings():  # đo cố ý vượt max → tắt warning spam
            warnings.simplefilter("ignore")
            lens = [len(x) for x in tokenizer(batch, padding=False, truncation=False)["input_ids"]]
        total_trunc += sum(1 for L in lens if L > max_length)
    return {"n": total, "n_truncated": total_trunc, "rate": total_trunc / max(1, total)}


# ---------------------------------------------------------------- train
def run_batches(model, tokenizer, refs, codes, max_length, batch_size, device, optimizer=None,
                accum_steps: int = 1, max_grad_norm: float | None = None):
    """Một pass: train (optimizer khác None) hoặc eval. Trả về loss/s/scores/labels/n_trunc."""
    train = optimizer is not None
    model.train(train)
    total_loss, n_trunc = 0.0, 0
    all_s, all_y = [], []
    criterion = nn.BCEWithLogitsLoss()
    use_amp = train and device.type == "cuda" and _amp_enabled()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if train:
        optimizer.zero_grad()
    n_micro = 0
    for i in range(0, len(refs), batch_size):
        chunk = refs[i : i + batch_size]
        ba, bb, y, t = collate_pair_batch(
            tokenizer, [codes[r["a"]] for r in chunk], [codes[r["b"]] for r in chunk],
            [r["label"] for r in chunk], max_length, device,
        )
        n_trunc += t
        if train:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ba, bb)
                loss = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()
            n_micro += 1
            total_loss += loss.item() * accum_steps * len(chunk)
            if n_micro % accum_steps == 0:
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                logits = model(ba, bb)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(chunk)
        all_s.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(np.array([r["label"] for r in chunk]))
    if train and n_micro % accum_steps != 0:  # phần dư cuối epoch
        if max_grad_norm:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return total_loss / max(1, len(refs)), np.concatenate(all_s), np.concatenate(all_y), n_trunc


_AMP = {"enabled": False}


def _amp_enabled() -> bool:
    return _AMP["enabled"]


def train_model(cfg, out_dir: Path, device: torch.device, max_pairs: int = 0, epochs: int = 0) -> dict:
    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    epochs = epochs or tr["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    model = CodeBERTPairClassifier(cfg["model"]["hf_id"], tr.get("dropout", 0.1)).to(device)
    max_len = cfg["tokenizer"]["max_length"]

    train_codes = load_code_map(processed, ["train"])
    train_refs = load_pair_refs(pairs_dir / "train.jsonl")
    val_refs = load_pair_refs(pairs_dir / "validation.jsonl")
    if max_pairs:  # smoke: subset cân bằng pos/neg, deterministic
        def balanced(refs, n):
            pos = [r for r in refs if r["label"] == 1][: n // 2]
            neg = [r for r in refs if r["label"] == 0][: n // 2]
            return pos + neg

        train_refs = balanced(train_refs, max_pairs)
        val_refs = balanced(val_refs, max_pairs // 2)
    val_codes = load_code_map(processed, ["validation"])

    trunc = {
        "train": truncation_analysis(tokenizer, train_codes, max_len),
        "validation": truncation_analysis(tokenizer, val_codes, max_len),
    }

    steps_per_epoch = max(1, len(train_refs) // (tr["batch_size"] * tr["gradient_accumulation_steps"]))
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * tr.get("warmup_ratio", 0.1)), total_steps
    )
    _AMP["enabled"] = bool(tr.get("mixed_precision", False)) and device.type == "cuda"

    es = cfg.get("early_stopping", {})
    best_f1, best_epoch, patience_left = -1.0, -1, int(es.get("patience", 2))
    history, global_step = [], 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        rng = random.Random(f"{tr['seed']}/{epoch}")
        rng.shuffle(train_refs)
        loss, _, _, n_tr = run_batches(
            model, tokenizer, train_refs, train_codes, max_len, tr["batch_size"], device, optimizer,
            accum_steps=tr["gradient_accumulation_steps"],
            max_grad_norm=tr.get("max_grad_norm"),
        )
        scheduler.step()
        global_step += steps_per_epoch
        v_loss, v_s, v_y, _ = run_batches(
            model, tokenizer, val_refs, val_codes, max_len, tr["batch_size"] * 2, device
        )
        vm = split_metrics(v_s, v_y, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": v_loss,
                        "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "lr": lr})
        ckpt = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "epoch": epoch,
                "global_step": global_step, "val_f1": vm["f1"]}
        torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch}.pt")
        improved = vm["f1"] > best_f1
        if improved:
            best_f1, best_epoch = vm["f1"], epoch
            torch.save(ckpt, out_dir / "best_checkpoint.pt")
            patience_left = int(es.get("patience", 2))
        else:
            patience_left -= 1
        print(json.dumps({"epoch": epoch, "train_loss": round(loss, 4),
                          "val_f1": round(vm["f1"], 4), "best_f1": round(best_f1, 4)}))
        if es.get("enabled", False) and patience_left < 0:
            print(f"early stopping tại epoch {epoch}")
            break
    train_time = time.perf_counter() - t0
    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    (out_dir / "truncation.json").write_text(json.dumps(trunc, indent=2), encoding="utf-8")
    meta = run_metadata(Path(cfg["_config_path"]), manifest_path)
    meta.update({"training_time_s": train_time, "best_epoch": best_epoch,
                 "best_val_f1": best_f1, "device_requested": str(device)})
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history,
            "truncation": trunc, "train_time_s": train_time}


# ---------------------------------------------------------------- evaluate
def evaluate_model(cfg, out_dir: Path, device: torch.device) -> dict:
    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    ckpt_path = out_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Thiếu best checkpoint: {ckpt_path} (§23)")
    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    model = CodeBERTPairClassifier(cfg["model"]["hf_id"]).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    max_len, bs = cfg["tokenizer"]["max_length"], cfg["inference"]["pair_batch_size"]

    scored = {}
    for split in ("validation", "test"):
        codes = load_code_map(processed, [split])
        refs = load_pair_refs(pairs_dir / f"{split}.jsonl")
        _, s, y, _ = run_batches(model, tokenizer, refs, codes, max_len, bs, device)
        scored[split] = (s, y, refs)
    thr = select_threshold(*scored["validation"][:2], step=cfg["threshold"]["grid_step"])
    threshold = thr["threshold"]
    results = {sp: split_metrics(s, y, threshold) for sp, (s, y, _) in scored.items()}

    top_k = cfg["error_analysis"]["top_k"]
    s, y, refs = scored["test"]
    order = np.argsort(s)
    err = []
    for idx in order[: len(s)]:  # FN: label 1 score thấp
        if len([e for e in err if e["error_type"] == "false_negative"]) >= top_k:
            break
        if y[idx] == 1 and s[idx] < threshold:
            r = refs[idx]
            err.append({"pair_id": r["pair_id"], "sample_id_a": r["a"], "sample_id_b": r["b"],
                        "problem_id_a": r["problem_a"], "problem_id_b": r["problem_b"],
                        "label": 1, "prediction": 0, "probability": float(s[idx]),
                        "error_type": "false_negative"})
    for idx in order[::-1]:  # FP: label 0 score cao
        if len([e for e in err if e["error_type"] == "false_positive"]) >= top_k:
            break
        if y[idx] == 0 and s[idx] >= threshold:
            r = refs[idx]
            err.append({"pair_id": r["pair_id"], "sample_id_a": r["a"], "sample_id_b": r["b"],
                        "problem_id_a": r["problem_a"], "problem_id_b": r["problem_b"],
                        "label": 0, "prediction": 1, "probability": float(s[idx]),
                        "error_type": "false_positive"})

    t0 = time.perf_counter()
    (out_dir / "threshold.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TP", "TN", "FP", "FN"]
        w.writerow(["split"] + keys)
        for sp, m in results.items():
            w.writerow([sp] + [m[k] for k in keys])
    (out_dir / "confusion_matrix.json").write_text(
        json.dumps({sp: {k: m[k] for k in ("TP", "TN", "FP", "FN")} for sp, m in results.items()}, indent=2),
        encoding="utf-8")
    with open(out_dir / "error_cases.jsonl", "w", encoding="utf-8") as f:
        for e in err:
            f.write(json.dumps(e) + "\n")
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for sp, (s, y, refs) in scored.items():
        with open(pred_dir / f"{sp}_scores.jsonl", "w", encoding="utf-8") as f:
            for r, v, t in zip(refs, s, y):
                f.write(json.dumps({"pair_id": r["pair_id"], "label": int(t), "score": float(v)}) + "\n")
    (out_dir / "model_stats.json").write_text(json.dumps({
        "hf_id": cfg["model"]["hf_id"], "hidden_size": model.hidden_size,
        "param_count": model.param_count(),
        "pair_architecture": "[hA; hB; |hA-hB|] + MLP(Tanh) — DEC-016",
        "max_length": max_len,
    }, indent=2), encoding="utf-8")
    eval_time = time.perf_counter() - t0
    summary = {"threshold": threshold, "validation_f1_at_thr": thr["f1"],
               "test": results["test"], "eval_time_s": eval_time, "out_dir": str(out_dir)}
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- preflight
def preflight(cfg, device: torch.device) -> dict:
    import transformers

    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "device_requested": str(device),
    }
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FAIL FAST: yêu cầu cuda nhưng torch.cuda.is_available() == False (§16)")
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "vram_free_gb": round(torch.cuda.mem_get_info(0)[0] / 1e9, 2),
        })
    model = CodeBERTPairClassifier(cfg["model"]["hf_id"])
    info.update({
        "hf_id": cfg["model"]["hf_id"],
        "param_count": model.param_count(),
        "hidden_size": model.hidden_size,
        "max_length": cfg["tokenizer"]["max_length"],
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "mixed_precision": cfg["training"].get("mixed_precision", False),
        "threshold_method": cfg["threshold"]["method"],
    })
    n_train = sum(1 for _ in stream_jsonl(Path(cfg["dataset"]["pairs_dir"]) / "train.jsonl"))
    info["n_train_pairs"] = n_train
    del model
    print(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["preflight", "smoke", "train", "evaluate"], required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = str(args.config)
    device = torch.device(args.device)
    out_dir = Path(cfg["output"]["directory"])

    if args.mode == "preflight":
        preflight(cfg, device)
    elif args.mode == "smoke":
        # Smoke CPU/GPU: 1 epoch, subset nhỏ — verify forward/backward/loss/checkpoint (§22).
        res = train_model(cfg, out_dir / "smoke", device, max_pairs=SMOKE_TRAIN_PAIRS, epochs=1)
        assert res["history"][0]["train_loss"] < float("inf")
        assert (out_dir / "smoke" / "best_checkpoint.pt").exists()
        print("SMOKE PASS")
    elif args.mode == "train":
        res = train_model(cfg, out_dir, device)
        (out_dir / "config.yaml").write_text(
            Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
        shutil.copy(Path(cfg["dataset"]["manifest"]), out_dir / "manifest.json")
        evaluate_model(cfg, out_dir, device)
        print(json.dumps({"best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"]}))
    elif args.mode == "evaluate":
        evaluate_model(cfg, out_dir, device)


if __name__ == "__main__":
    main()
