"""EXP-002 — M1 CodeBERT fine-tuning trên POJ-104.

Chạy (trên máy GPU của user):
  preflight:  python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml --mode preflight --device cuda
  diagnostic: ... --mode diagnostic --device cuda
  smoke:      ... --mode smoke --device cuda
  train:      ... --mode train --device cuda [--time-budget-s 32400]
  evaluate:   ... --mode evaluate --device cuda
  resume:     ... --mode resume --device cuda   (nạp last_checkpoint.pt, train tiếp)

Protocol (§35): train chỉ update trên train; val chỉ eval; test untouched tới
final evaluation; threshold trên val (F1); best checkpoint theo val F1.

Chạy trên Kaggle (DEC-018): phiên GPU có giới hạn 12h và bị kill không báo
trước. Vì vậy mọi artifact (training_history.csv, run_metadata.json,
last/best_checkpoint.pt) được flush ngay sau TỪNG epoch, `--time-budget-s`
dừng train graceful để kịp chạy evaluate, và `--mode resume` cho phép
train tiếp thay vì chạy lại từ đầu.
"""

import argparse
import csv
import json
import random
import shutil
import sys
import time
from datetime import datetime
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

# Ngân sách thời gian mặc định cho 1 phiên Kaggle (12h). Reserve ~2.5h cho
# setup + eval + tar + margin để notebook không bị kill giữa chừng.
DEFAULT_TIME_BUDGET_S = 9 * 3600


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


def _restrict_codes(codes: dict[str, str], refs: list[dict]) -> dict[str, str]:
    """Chỉ giữ code mà `refs` thực sự tham chiếu (dùng cho smoke/diagnostic)."""
    need = {r["a"] for r in refs} | {r["b"] for r in refs}
    return {k: v for k, v in codes.items() if k in need}


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
def _optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm):
    """Một optimizer step duy nhất + scheduler step (per-update, DEC-017)."""
    if max_grad_norm:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    if scheduler is not None:
        scheduler.step()  # FIX DEC-017: step mỗi optimizer update, không phải mỗi epoch
    return 1


def _assert_device_match(device: torch.device, *tensors, name: str = "") -> None:
    """Fail-fast: mọi tensor phải nằm đúng device yêu cầu."""
    for t in tensors:
        if t.device.type != device.type:
            raise RuntimeError(
                f"FAIL FAST: tensor {name} trên {t.device}, kỳ vọng {device} (§7)"
            )


def run_batches(model, tokenizer, refs, codes, max_length, batch_size, device, optimizer=None,
                accum_steps: int = 1, max_grad_norm: float | None = None,
                scheduler=None):
    """Một pass: train (optimizer khác None) hoặc eval. Trả về loss/s/scores/labels/n_trunc/opt_steps."""
    train = optimizer is not None
    model.train(train)
    total_loss, n_trunc, opt_steps = 0.0, 0, 0
    all_s, all_y = [], []
    criterion = nn.BCEWithLogitsLoss()
    use_amp = train and device.type == "cuda" and _amp_enabled()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if train:
        optimizer.zero_grad()
    import sys as _sys
    import time as _time
    n_micro = 0
    _t_start = _time.perf_counter()
    total_chunks = max(1, (len(refs) + batch_size - 1) // batch_size)
    log_every = max(1, total_chunks // 20)  # ~5% mỗi dòng log
    for ci, i in enumerate(range(0, len(refs), batch_size)):
        chunk = refs[i : i + batch_size]
        ba, bb, y, t = collate_pair_batch(
            tokenizer, [codes[r["a"]] for r in chunk], [codes[r["b"]] for r in chunk],
            [r["label"] for r in chunk], max_length, device,
        )
        if device.type == "cuda":
            for d in (ba, bb):
                for v in d.values():
                    _assert_device_match(device, v, name="input")
            _assert_device_match(device, y, name="labels")
        n_trunc += t
        if train:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ba, bb)
                if logits.device.type != device.type:
                    raise RuntimeError(
                        f"FAIL FAST: logits trên {logits.device}, kỳ vọng {device} (§7)"
                    )
                loss = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()
            n_micro += 1
            total_loss += loss.item() * accum_steps * len(chunk)
            if n_micro % accum_steps == 0:
                opt_steps += _optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm)
        else:
            with torch.no_grad():
                logits = model(ba, bb)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(chunk)
        all_s.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(np.array([r["label"] for r in chunk]))
        if (ci + 1) % log_every == 0 or (ci + 1) == total_chunks:
            phase = "train" if train else "eval"
            _el = _time.perf_counter() - _t_start
            _rate = (ci + 1) / max(1e-9, _el)
            _eta = (total_chunks - (ci + 1)) / max(1e-9, _rate)


            def _fmt(s: float) -> str:
                s = int(s)
                h, s = divmod(s, 3600)
                m, s = divmod(s, 60)
                return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

            print(f"[{phase}] {(ci + 1) / total_chunks * 100:5.1f}% "
                  f"({ci + 1}/{total_chunks}) loss_so_far={total_loss / max(1, i + len(chunk)):.4f} "
                  f"elapsed={_fmt(_el)} eta={_fmt(_eta)}",
                  flush=True)
            _sys.stdout.flush()
    if train and n_micro % accum_steps != 0:  # phần dư cuối epoch
        opt_steps += _optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm)
    return total_loss / max(1, len(refs)), np.concatenate(all_s), np.concatenate(all_y), n_trunc, opt_steps


_AMP = {"enabled": False}


def _amp_enabled() -> bool:
    return _AMP["enabled"]


# ---------------------------------------------------------------- durable artifacts
# Kaggle kill phiên không báo trước. Mọi artifact phải flush ngay sau mỗi epoch
# (ghi atomic qua file.tmp + replace) để không mất sạch khi session chết.
def _atomic_write_text(path: Path, text: str) -> None:
    # newline="" để không dịch \n -> \r\n trên Windows; nếu không, output của
    # csv.writer (\r\n) sẽ thành \r\r\n và sinh dòng rỗng lẫn giữa file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    tmp.replace(path)


def write_history(out_dir: Path, history: list[dict]) -> None:
    if not history:
        return
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(history[0].keys()))
    w.writeheader()
    w.writerows(history)
    _atomic_write_text(out_dir / "training_history.csv", buf.getvalue())


def write_run_metadata(out_dir: Path, cfg: dict, manifest_path: Path, device, model) -> dict:
    meta = run_metadata(Path(cfg["_config_path"]), manifest_path)
    meta.update(collect_device_metadata(device, model))
    meta.update({
        "scheduler": "linear_warmup_per_update_step (DEC-017)",
        "wall_clock": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return meta


def save_training_state(out_dir: Path, model, optimizer, scheduler, epoch: int,
                        global_step: int, best_f1: float, best_epoch: int,
                        is_best: bool, val_f1: float) -> None:
    """Chỉ giữ `last` + `best` — bỏ checkpoint từng epoch để tiết kiệm ~3GB
    trong /kaggle/working (giới hạn 20GB)."""
    ckpt = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "epoch": epoch,
            "global_step": global_step, "best_f1": best_f1, "best_epoch": best_epoch,
            "val_f1": val_f1,
            "rng": {"python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp = out_dir / "last_checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.replace(out_dir / "last_checkpoint.pt")
    if is_best:
        tmp2 = out_dir / "best_checkpoint.pt.tmp"
        torch.save(ckpt, tmp2)
        tmp2.replace(out_dir / "best_checkpoint.pt")


def load_training_state(out_dir: Path, model, optimizer, scheduler) -> dict:
    """Nạp `last_checkpoint.pt` để resume. Raise nếu không tồn tại (fail-fast)."""
    path = out_dir / "last_checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"Không có checkpoint để resume: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])
    rng = ckpt.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return ckpt


def train_model(cfg, out_dir: Path, device: torch.device, max_pairs: int = 0, epochs: int = 0,
                time_budget_s: int = 0, resume: bool = False) -> dict:
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
    assert_device_setup(device, model)  # FAIL FAST nếu CPU fallback (§7)
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

    if max_pairs:
        # Smoke chỉ cần đúng các code nó dùng. Giữ toàn bộ 41k codes sẽ tốn
        # ~4 phút cho truncation_analysis dù chỉ forward 256 pairs.
        train_codes = _restrict_codes(train_codes, train_refs)
        val_codes = _restrict_codes(val_codes, val_refs)

    trunc_path = out_dir / "truncation.json"
    if resume and trunc_path.exists():
        trunc = json.loads(trunc_path.read_text(encoding="utf-8"))  # ~4 phút, bỏ lại khi resume
        print("resume: dùng lại truncation.json")
    else:
        trunc = {
            "train": truncation_analysis(tokenizer, train_codes, max_len),
            "validation": truncation_analysis(tokenizer, val_codes, max_len),
        }
        _atomic_write_text(trunc_path, json.dumps(trunc, indent=2))

    import math
    # Số optimizer updates thực tế mỗi epoch (ceil: tính cả batch dư cuối).
    steps_per_epoch = max(1, math.ceil(len(train_refs) / tr["batch_size"] / tr["gradient_accumulation_steps"]))
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * tr.get("warmup_ratio", 0.1)), total_steps
    )
    _AMP["enabled"] = bool(tr.get("mixed_precision", False)) and device.type == "cuda"

    es = cfg.get("early_stopping", {})
    best_f1, best_epoch, patience_left = -1.0, -1, int(es.get("patience", 2))
    history, global_step = [], 0
    start_epoch = 0

    if resume:
        ckpt = load_training_state(out_dir, model, optimizer, scheduler)
        start_epoch = int(ckpt["epoch"]) + 1
        best_f1, best_epoch = float(ckpt["best_f1"]), int(ckpt["best_epoch"])
        global_step = int(ckpt["global_step"])
        hist_path = out_dir / "training_history.csv"
        if hist_path.exists():
            with open(hist_path, newline="", encoding="utf-8") as f:
                history = [r for r in csv.DictReader(f)]
        print(json.dumps({"RESUME": True, "from_epoch": start_epoch, "best_f1": best_f1,
                          "best_epoch": best_epoch, "global_step": global_step}))
        if start_epoch >= epochs:
            print(json.dumps({"WARNING": "đã train đủ epoch, chỉ cần evaluate"}))

    t0 = time.perf_counter()
    prev_elapsed = 0.0
    stopped_reason = "epochs_exhausted"
    for epoch in range(start_epoch, epochs):
        rng = random.Random(f"{tr['seed']}/{epoch}")
        rng.shuffle(train_refs)
        loss, _, _, n_tr, opt_steps = run_batches(
            model, tokenizer, train_refs, train_codes, max_len, tr["batch_size"], device, optimizer,
            accum_steps=tr["gradient_accumulation_steps"],
            max_grad_norm=tr.get("max_grad_norm"),
            scheduler=scheduler,  # DEC-017: scheduler step mỗi update trong run_batches
        )
        global_step += opt_steps
        v_loss, v_s, v_y, _, _ = run_batches(
            model, tokenizer, val_refs, val_codes, max_len, tr["batch_size"] * 2, device
        )
        vm = split_metrics(v_s, v_y, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": v_loss,
                        "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "lr": lr,
                        "epoch_time_s": round(elapsed - prev_elapsed, 1),
                        "elapsed_s": round(elapsed, 1),
                        "global_step": global_step})
        prev_elapsed = elapsed

        improved = vm["f1"] > best_f1
        if improved:
            best_f1, best_epoch = vm["f1"], epoch
            patience_left = int(es.get("patience", 2))
        else:
            patience_left -= 1
        save_training_state(out_dir, model, optimizer, scheduler, epoch, global_step,
                            best_f1, best_epoch, improved, vm["f1"])
        write_history(out_dir, history)  # flush ngay — không đợi hết epoch
        write_run_metadata(out_dir, cfg, manifest_path, device, model)
        print(json.dumps({"epoch": epoch, "train_loss": round(loss, 4),
                          "val_f1": round(vm["f1"], 4), "best_f1": round(best_f1, 4),
                          "epoch_time_s": history[-1]["epoch_time_s"],
                          "elapsed_s": history[-1]["elapsed_s"]}))

        if es.get("enabled", False) and patience_left < 0:
            stopped_reason = "early_stopping"
            print(f"early stopping tại epoch {epoch}")
            break
        if time_budget_s and elapsed > time_budget_s:
            stopped_reason = "time_budget"
            print(json.dumps({"WARNING": "time budget reached — dừng train, chuyển sang evaluate",
                              "elapsed_s": round(elapsed, 1), "budget_s": time_budget_s}))
            break

    train_time = time.perf_counter() - t0
    write_history(out_dir, history)
    trunc = json.loads(trunc_path.read_text(encoding="utf-8"))
    meta = write_run_metadata(out_dir, cfg, manifest_path, device, model)
    meta.update({"training_time_s": train_time, "best_epoch": best_epoch,
                 "best_val_f1": best_f1, "total_optimizer_steps": global_step,
                 "stopped_reason": stopped_reason, "epochs_completed": len(history),
                 "time_budget_s": time_budget_s or None})
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history,
            "truncation": trunc, "train_time_s": train_time,
            "stopped_reason": stopped_reason, "epochs_completed": len(history)}


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
    assert_device_setup(device, model)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    max_len, bs = cfg["tokenizer"]["max_length"], cfg["inference"]["pair_batch_size"]

    scored = {}
    for split in ("validation", "test"):
        codes = load_code_map(processed, [split])
        refs = load_pair_refs(pairs_dir / f"{split}.jsonl")
        _, s, y, _, _ = run_batches(model, tokenizer, refs, codes, max_len, bs, device)
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
    _atomic_write_text(out_dir / "threshold.json", json.dumps(thr, indent=2))
    _atomic_write_text(out_dir / "metrics.json", json.dumps(results, indent=2))
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TP", "TN", "FP", "FN"]
        w.writerow(["split"] + keys)
        for sp, m in results.items():
            w.writerow([sp] + [m[k] for k in keys])
    _atomic_write_text(out_dir / "confusion_matrix.json", json.dumps(
        {sp: {k: m[k] for k in ("TP", "TN", "FP", "FN")} for sp, m in results.items()}, indent=2))
    with open(out_dir / "error_cases.jsonl", "w", encoding="utf-8") as f:
        for e in err:
            f.write(json.dumps(e) + "\n")
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for sp, (s, y, refs) in scored.items():
        with open(pred_dir / f"{sp}_scores.jsonl", "w", encoding="utf-8") as f:
            for r, v, t in zip(refs, s, y):
                f.write(json.dumps({"pair_id": r["pair_id"], "label": int(t), "score": float(v)}) + "\n")
    _atomic_write_text(out_dir / "model_stats.json", json.dumps({
        "hf_id": cfg["model"]["hf_id"], "hidden_size": model.hidden_size,
        "param_count": model.param_count(),
        "pair_architecture": "[hA; hB; |hA-hB|] + MLP(Tanh) — DEC-016",
        "max_length": max_len,
    }, indent=2))
    eval_time = time.perf_counter() - t0
    summary = {"threshold": threshold, "validation_f1_at_thr": thr["f1"],
               "test": results["test"], "eval_time_s": eval_time, "out_dir": str(out_dir)}
    # Marker: notebook dùng để biết chắc eval đã xong trước khi tar.
    _atomic_write_text(out_dir / "EVAL_COMPLETE.json",
                       json.dumps({"completed": True, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary



# ---------------------------------------------------------------- device
def assert_device_setup(device: torch.device, model: nn.Module) -> None:
    """FAIL FAST nếu --device cuda nhưng GPU không thực sự dùng (§6–§7)."""
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL FAST: --device cuda nhưng torch.cuda.is_available() == False (§7)")
        mdev = next(model.parameters()).device
        if mdev.type != "cuda":
            raise RuntimeError(f"FAIL FAST: model trên {mdev}, kỳ vọng cuda (§7)")


def collect_device_metadata(device: torch.device, model: nn.Module) -> dict:
    """Metadata runtime thực tế (§8) — thay thế device='cpu' hardcode."""
    mdev = str(next(model.parameters()).device)
    meta = {
        "device_requested": str(device),
        "device_actual": mdev,
        "model_device": mdev,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        meta.update({
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "gpu_memory_allocated": torch.cuda.memory_allocated(0),
            "gpu_memory_reserved": torch.cuda.memory_reserved(0),
        })
    return meta


# ---------------------------------------------------------------- diagnostic
DIAG_TRAIN_PAIRS = 64
DIAG_VAL_PAIRS = 32


def diagnostic(cfg, device: torch.device) -> dict:
    """Diagnostic nhỏ trên GPU: verify device/gradients/param-update/logit variance (§9–§10, §33–§34).

    KHÔNG phải kết quả paper. PASS = implementation đúng, không yêu cầu F1 cụ thể.
    """
    from torch.optim import AdamW

    print(json.dumps({
        "CUDA available": torch.cuda.is_available(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FAIL FAST: diagnostic yêu cầu cuda nhưng không có GPU (§10)")

    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    max_len = cfg["tokenizer"]["max_length"]

    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    model = CodeBERTPairClassifier(cfg["model"]["hf_id"]).to(device)
    assert_device_setup(device, model)
    mdev = str(next(model.parameters()).device)
    print(json.dumps({"Model device": mdev}))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_train = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(json.dumps({"trainable_params": trainable, "encoder_trainable": enc_train,
                      "classifier_trainable": head_train}))
    assert enc_train > 0 and head_train > 0, "FAIL: encoder hoặc classifier bị freeze toàn bộ"

    codes = load_code_map(processed, ["train"])
    refs = load_pair_refs(pairs_dir / "train.jsonl")
    pos = [r for r in refs if r["label"] == 1][: DIAG_TRAIN_PAIRS // 2]
    neg = [r for r in refs if r["label"] == 0][: DIAG_TRAIN_PAIRS // 2]
    diag_refs = pos + neg
    assert len(pos) and len(neg), "FAIL: diagnostic subset thiếu pos/neg"

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])
    _AMP["enabled"] = False  # diagnostic: tắt AMP cho rõ ràng (§29)
    losses = []
    for epoch in range(2):
        loss, s, y, _, _ = run_batches(
            model, tokenizer, diag_refs, codes, max_len, 8, device, optimizer,
            accum_steps=1, max_grad_norm=tr.get("max_grad_norm"),
        )
        losses.append(loss)
        uniq = sorted(set(int(v) for v in y))
        print(json.dumps({
            "epoch": epoch, "loss": loss, "labels_unique": uniq,
            "score_min": float(s.min()), "score_max": float(s.max()),
            "score_mean": float(s.mean()), "score_std": float(s.std()),
            "input_device": "cuda" if device.type == "cuda" else "cpu",
        }))

    # Gradient check trên một micro-batch riêng (run_batches đã zero_grad cuối epoch).
    import torch.nn as _nn
    model.train(True)
    optimizer.zero_grad()
    _chunk = diag_refs[:8]
    _ba, _bb, _y, _ = collate_pair_batch(
        tokenizer, [codes[r["a"]] for r in _chunk], [codes[r["b"]] for r in _chunk],
        [r["label"] for r in _chunk], max_len, device)
    _loss = _nn.BCEWithLogitsLoss()(model(_ba, _bb), _y)
    _loss.backward()
    grads_ok, grad_norm, n_with_grad = True, 0.0, 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            n_with_grad += 1
            if not torch.isfinite(p.grad).all():
                grads_ok = False
            grad_norm += float(p.grad.norm())
    if n_with_grad == 0:
        grads_ok = False
    optimizer.zero_grad()
    changed = sum(1 for n, p in model.named_parameters()
                  if p.requires_grad and not torch.equal(before[n].cpu(), p.detach().cpu()))
    score_std = float(s.std())
    checks = {
        "cuda_confirmed": device.type != "cuda" or torch.cuda.is_available(),
        "model_on_cuda": mdev.startswith("cuda") if device.type == "cuda" else True,
        "pretrained_encoder": enc_train > 0,
        "gradients_finite": grads_ok and grad_norm > 0,
        "params_updated": changed > 0,
        "logits_vary": score_std > 0,
        "probs_not_constant": not (float(s.min()) == float(s.max()) == 0.5),
        "loss_finite": all(np.isfinite(losses)),
    }
    checks["grad_norm"] = grad_norm
    checks["params_changed"] = changed
    checks["losses"] = losses
    print(json.dumps({"DIAGNOSTIC": "PASS" if all(v for k, v in checks.items()
                    if isinstance(v, bool)) else "FAIL", **{k: v for k, v in checks.items()
                    if isinstance(v, bool)}}))
    if torch.cuda.is_available():
        print(json.dumps({"gpu_memory_allocated": torch.cuda.memory_allocated(0),
                          "gpu_memory_reserved": torch.cuda.memory_reserved(0)}))
    return checks


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
    ap.add_argument("--mode", choices=["preflight", "smoke", "train", "evaluate",
                                       "diagnostic", "resume"], required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--time-budget-s", type=int, default=DEFAULT_TIME_BUDGET_S,
                    help="Dừng train gracefully sau N giây rồi chạy evaluate "
                         "(bảo vệ giới hạn phiên Kaggle 12h). 0 = tắt.")
    ap.add_argument("--resume", action="store_true",
                    help="Nạp last_checkpoint.pt và train tiếp các epoch còn lại.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["_config_path"] = str(args.config)
    device = torch.device(args.device)
    out_dir = Path(cfg["output"]["directory"])

    if args.mode == "preflight":
        preflight(cfg, device)
    elif args.mode == "diagnostic":
        diagnostic(cfg, device)
    elif args.mode == "smoke":
        # Smoke CPU/GPU: 1 epoch, subset nhỏ — verify forward/backward/loss/checkpoint (§22).
        res = train_model(cfg, out_dir / "smoke", device, max_pairs=SMOKE_TRAIN_PAIRS, epochs=1,
                          time_budget_s=0)
        assert res["history"][0]["train_loss"] < float("inf")
        assert (out_dir / "smoke" / "best_checkpoint.pt").exists()
        print("SMOKE PASS")
    elif args.mode in ("train", "resume"):
        resume = args.resume or args.mode == "resume"
        res = train_model(cfg, out_dir, device, time_budget_s=args.time_budget_s, resume=resume)
        _atomic_write_text(out_dir / "config.yaml", Path(args.config).read_text(encoding="utf-8"))
        shutil.copy(Path(cfg["dataset"]["manifest"]), out_dir / "manifest.json")
        evaluate_model(cfg, out_dir, device)
        print(json.dumps({"best_epoch": res["best_epoch"], "best_val_f1": res["best_val_f1"],
                          "stopped_reason": res["stopped_reason"],
                          "epochs_completed": res["epochs_completed"]}))
    elif args.mode == "evaluate":
        evaluate_model(cfg, out_dir, device)


if __name__ == "__main__":
    main()
