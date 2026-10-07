"""EXP-006 — M5 Hybrid (GraphCodeBERT + CPG-GNN): train + evaluate (POJ-104, C++).

Chạy trên Kaggle GPU, SAU notebook extraction (cần CPG cache như EXP-005):
    python -m src.experiments.exp006_hybrid --config configs/exp006_hybrid.yaml \
        --mode preflight --device cuda
    python -m src.experiments.exp006_hybrid --config ... --mode smoke ...
    python -m src.experiments.exp006_hybrid --config ... --mode train ...
    # train tự evaluate khi xong; --mode evaluate chạy riêng khi cần.

Dữ liệu vào: text (`data/processed/poj104/samples.jsonl`) + CPG (`artifacts/cpg_export`)
như EXP-002/003/005. Hợp đồng artifacts RA giống hệt các model khác
(threshold trên validation, test một lần): threshold.json, metrics.json/csv,
confusion_matrix.json, error_cases.jsonl, predictions/*_scores.jsonl,
model_stats.json, EVAL_COMPLETE.json, training_history.csv.

Khác biệt so với M1/M2/M4 ở **data path** (mỗi phía một biểu diễn); mọi tầng còn
lại (artifact bền vững, resume, early stopping, threshold) giữ nguyên để phép so
sánh ở RQ3 công bằng ở mức protocol. Cache CPG dùng lại đúng của EXP-005
(`cpp_{split}_full.npz`); không parse lại Joern.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.experiments.exp001_tfidf import (
    run_metadata, select_threshold, split_metrics, validate_dataset,
)
from src.experiments.exp005_cpg_gnn import (
    MAX_MISSING_PAIR_RATE, build_stores, filter_refs, required_ids,
)
from src.experiments.transformer_pair_runner import (
    DEFAULT_TIME_BUDGET_S, _AMP, _assert_device_match, _atomic_write_text,
    _optimizer_step, _restrict_codes, assert_device_setup, collect_device_metadata,
    load_code_map, load_eval_weights, load_pair_refs, load_training_state,
    resolve_eval_checkpoint, save_training_state, truncation_analysis,
    write_history,
)
from src.models.cpg_gnn.model import to_device as graph_to_device
from src.models.hybrid.model import (
    HybridPairClassifier, warm_start_graph, warm_start_text,
)
from src.models.transformer_pair import load_tokenizer, tokenize_side
from src.representations.cpg.schema import N_EDGE_TYPES

SMOKE_TRAIN_PAIRS = 64
SMOKE_VAL_PAIRS = 32


def build_model(cfg: dict, device: torch.device) -> HybridPairClassifier:
    mo = cfg["model"]
    model = HybridPairClassifier(
        hf_id=mo["hf_id"],
        graph_hidden=int(mo["graph_hidden_size"]),
        graph_num_layers=int(mo["graph_num_layers"]),
        graph_heads=int(mo["graph_heads"]),
        graph_dropout=float(mo.get("graph_dropout", 0.2)),
        head_hidden=int(mo.get("head_hidden_size", 1024)),
        head_dropout=float(mo.get("head_dropout", 0.1)),
        freeze_encoders=bool(cfg["training"].get("freeze_encoders", False)),
    ).to(device)
    return model


def collate_pair(tokenizer, codes: dict[str, str], store, chunk: list[dict],
                 max_length: int, device):
    ta, n_tr_a = tokenize_side(tokenizer, [codes[r["a"]] for r in chunk], max_length)
    tb, n_tr_b = tokenize_side(tokenizer, [codes[r["b"]] for r in chunk], max_length)
    ta = {k: v.to(device) for k, v in ta.items()}
    tb = {k: v.to(device) for k, v in tb.items()}
    ga = graph_to_device(store.collate([r["a"] for r in chunk]), device)
    gb = graph_to_device(store.collate([r["b"] for r in chunk]), device)
    y = torch.tensor([float(r["label"]) for r in chunk], dtype=torch.float32, device=device)
    return ta, ga, tb, gb, y, n_tr_a + n_tr_b


def _assert_text_on_device(device, batch: dict, name: str) -> None:
    for v in batch.values():
        _assert_device_match(device, v, name=name)


def _assert_graph_on_device(device, batch, name: str) -> None:
    for key in ("node_type", "x", "edge_index", "edge_type"):
        _assert_device_match(device, getattr(batch, key), name=f"{name}.{key}")


def run_batches(model, tokenizer, codes, store, refs, max_length, batch_size, device,
                optimizer=None, accum_steps: int = 1, max_grad_norm: float | None = None,
                scheduler=None):
    """Một pass: train (optimizer khác None) hoặc eval.

    Trả về `loss, scores, labels, n_nodes, n_text_trunc, opt_steps` — `n_nodes`
    tổng node đồ thị (báo card GAT), `n_text_trunc` số code text bị cắt máx_length.
    """
    train = optimizer is not None
    model.train(train)
    total_loss, n_nodes, n_text_trunc, opt_steps = 0.0, 0, 0, 0
    all_s, all_y = [], []
    criterion = nn.BCEWithLogitsLoss()
    use_amp = train and device.type == "cuda" and _AMP["enabled"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if train:
        optimizer.zero_grad()
    n_micro, seen = 0, 0
    t_start = time.perf_counter()
    total_chunks = max(1, (len(refs) + batch_size - 1) // batch_size)
    log_every = max(1, total_chunks // 20)
    for ci, i in enumerate(range(0, len(refs), batch_size)):
        chunk = refs[i : i + batch_size]
        ta, ga, tb, gb, y, n_tr = collate_pair(
            tokenizer, codes, store, chunk, max_length, device)
        if device.type == "cuda":
            _assert_text_on_device(device, ta, "input_a")
            _assert_text_on_device(device, tb, "input_b")
            _assert_graph_on_device(device, ga, "input_a")
            _assert_graph_on_device(device, gb, "input_b")
            _assert_device_match(device, y, name="labels")
        n_nodes += int(ga.node_type.shape[0]) + int(gb.node_type.shape[0])
        n_text_trunc += n_tr
        if train:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ta, ga, tb, gb)
                if logits.device.type != device.type:
                    raise RuntimeError(
                        f"FAIL FAST: logits trên {logits.device}, kỳ vọng {device}")
                loss = criterion(logits, y) / accum_steps
            scaler.scale(loss).backward()
            n_micro += 1
            total_loss += loss.item() * accum_steps * len(chunk)
            if n_micro % accum_steps == 0:
                opt_steps += _optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm)
        else:
            with torch.no_grad():
                logits = model(ta, ga, tb, gb)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(chunk)
        seen += len(chunk)
        all_s.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(np.array([r["label"] for r in chunk]))
        if (ci + 1) % log_every == 0 or (ci + 1) == total_chunks:
            el = time.perf_counter() - t_start
            rate = (ci + 1) / max(1e-9, el)
            eta = (total_chunks - (ci + 1)) / max(1e-9, rate)
            print(f"[{'train' if train else 'eval'}] "
                  f"{(ci + 1) / total_chunks * 100:5.1f}% ({ci + 1}/{total_chunks}) "
                  f"loss_so_far={total_loss / max(1, seen):.4f} "
                  f"elapsed={_fmt(el)} eta={_fmt(eta)}", flush=True)
    if train and n_micro % accum_steps != 0:
        opt_steps += _optimizer_step(model, optimizer, scaler, scheduler, max_grad_norm)
    return (total_loss / max(1, len(refs)), np.concatenate(all_s), np.concatenate(all_y),
            n_nodes, n_text_trunc, opt_steps)


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def write_run_metadata(out_dir: Path, cfg: dict, manifest_path: Path, device, model) -> dict:
    meta = run_metadata(Path(cfg["_config_path"]), manifest_path)
    meta.update(collect_device_metadata(device, model))
    meta.update({
        "scheduler": "linear_warmup_per_update_step (DEC-017)",
        "wall_clock": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return meta


def balanced_refs(refs: list[dict], n: int) -> list[dict]:
    return ([r for r in refs if r["label"] == 1][: n // 2]
            + [r for r in refs if r["label"] == 0][: n // 2])


# ---------------------------------------------------------------------- train
def train_model(cfg, out_dir: Path, device: torch.device, max_pairs: int = 0,
                epochs: int = 0, time_budget_s: int = 0, resume: bool = False) -> dict:
    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    epochs = epochs or tr["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    model = build_model(cfg, device)
    assert_device_setup(device, model)
    print(json.dumps({"param_count": model.param_count(),
                      "trainable": model.trainable_param_count(),
                      "text_dim": model.text_dim, "graph_dim": model.graph_dim,
                      "edge_types": N_EDGE_TYPES}))

    if not model.freeze_encoders:
        _warm_start(cfg, model, device)

    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    max_len = int(cfg["tokenizer"]["max_length"])

    train_refs = load_pair_refs(pairs_dir / "train.jsonl")
    val_refs = load_pair_refs(pairs_dir / "validation.jsonl")
    if max_pairs:
        train_refs = balanced_refs(train_refs, max_pairs)
        val_refs = balanced_refs(val_refs, max_pairs // 2)
    codes = load_code_map(processed, ["train", "validation"])
    codes = _restrict_codes(codes, train_refs + val_refs)

    stores = build_stores(cfg, pairs_dir, ["train", "validation"])
    train_refs = filter_refs(stores["train"], train_refs, "train")
    val_refs = filter_refs(stores["validation"], val_refs, "validation")
    train_store, val_store = stores["train"], stores["validation"]

    trunc = {"train": truncation_analysis(tokenizer,
                                          _restrict_codes(codes, train_refs), max_len),
             "validation": truncation_analysis(tokenizer,
                                               _restrict_codes(codes, val_refs), max_len)}
    _atomic_write_text(out_dir / "truncation.json", json.dumps(trunc, indent=2))

    bs = int(tr["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_refs) / bs / tr["gradient_accumulation_steps"]))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=tr["learning_rate"], weight_decay=tr.get("weight_decay", 0.0))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * tr.get("warmup_ratio", 0.1)), total_steps)
    _AMP["enabled"] = bool(tr.get("mixed_precision", False)) and device.type == "cuda"

    es = cfg.get("early_stopping", {})
    best_f1, best_epoch, patience_left = -1.0, -1, int(es.get("patience", 2))
    history, global_step, start_epoch = [], 0, 0

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

    t0 = time.perf_counter()
    prev_elapsed = 0.0
    stopped_reason = "epochs_exhausted"
    for epoch in range(start_epoch, epochs):
        random.Random(f"{tr['seed']}/{epoch}").shuffle(train_refs)
        loss, _, _, n_nodes, n_trunc, opt_steps = run_batches(
            model, tokenizer, codes, train_store, train_refs, max_len, bs, device,
            optimizer, accum_steps=tr["gradient_accumulation_steps"],
            max_grad_norm=tr.get("max_grad_norm"), scheduler=scheduler)
        global_step += opt_steps
        v_loss, v_s, v_y, _, _, _ = run_batches(
            model, tokenizer, codes, val_store, val_refs, max_len, bs * 2, device)
        vm = split_metrics(v_s, v_y, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": v_loss,
                        "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "lr": lr,
                        "n_nodes_train": n_nodes, "n_trunc_train": n_trunc,
                        "epoch_time_s": round(elapsed - prev_elapsed, 1),
                        "elapsed_s": round(elapsed, 1), "global_step": global_step})
        prev_elapsed = elapsed

        improved = vm["f1"] > best_f1
        if improved:
            best_f1, best_epoch = vm["f1"], epoch
            patience_left = int(es.get("patience", 2))
        else:
            patience_left -= 1
        save_training_state(out_dir, model, optimizer, scheduler, epoch, global_step,
                            best_f1, best_epoch, improved, vm["f1"])
        write_history(out_dir, history)
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
    meta = write_run_metadata(out_dir, cfg, manifest_path, device, model)
    meta.update({"training_time_s": train_time, "best_epoch": best_epoch,
                 "best_val_f1": best_f1, "total_optimizer_steps": global_step,
                 "stopped_reason": stopped_reason, "epochs_completed": len(history),
                 "freeze_encoders": bool(model.freeze_encoders),
                 "time_budget_s": time_budget_s or None})
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history,
            "train_time_s": train_time, "stopped_reason": stopped_reason,
            "epochs_completed": len(history)}


def _warm_start(cfg: dict, model: HybridPairClassifier, device: torch.device) -> None:
    ws = cfg["model"].get("warm_start", {}) or {}
    if ws.get("text"):
        warm_start_text(model.text_encoder, ws["text"], device)
    if ws.get("graph"):
        warm_start_graph(model.graph_encoder, ws["graph"], device)


# ------------------------------------------------------------------- evaluate
def evaluate_model(cfg, out_dir: Path, device: torch.device) -> dict:
    t_start = time.perf_counter()
    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    ckpt_path = resolve_eval_checkpoint(out_dir)
    model = build_model(cfg, device)
    assert_device_setup(device, model)
    model.load_state_dict(load_eval_weights(ckpt_path, device))
    model.eval()
    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    max_len = int(cfg["tokenizer"]["max_length"])
    bs = int(cfg["inference"]["pair_batch_size"])
    codes = load_code_map(processed, ["validation", "test"])

    stores = build_stores(cfg, pairs_dir, ["validation", "test"])
    scored = {}
    for split in ("validation", "test"):
        refs = filter_refs(stores[split], load_pair_refs(pairs_dir / f"{split}.jsonl"), split)
        codes_split = _restrict_codes(codes, refs)
        _, s, y, _, _, _ = run_batches(model, tokenizer, codes_split, stores[split],
                                       refs, max_len, bs, device)
        scored[split] = (s, y, refs)

    thr = select_threshold(*scored["validation"][:2], step=cfg["threshold"]["grid_step"])
    threshold = thr["threshold"]
    results = {sp: split_metrics(s, y, threshold) for sp, (s, y, _) in scored.items()}

    top_k = cfg["error_analysis"]["top_k"]
    s, y, refs = scored["test"]
    order = np.argsort(s)
    err = []
    for idx in order[: len(s)]:
        if len([e for e in err if e["error_type"] == "false_negative"]) >= top_k:
            break
        if y[idx] == 1 and s[idx] < threshold:
            r = refs[idx]
            err.append({"pair_id": r["pair_id"], "sample_id_a": r["a"], "sample_id_b": r["b"],
                        "problem_id_a": r["problem_a"], "problem_id_b": r["problem_b"],
                        "label": 1, "prediction": 0, "probability": float(s[idx]),
                        "error_type": "false_negative"})
    for idx in order[::-1]:
        if len([e for e in err if e["error_type"] == "false_positive"]) >= top_k:
            break
        if y[idx] == 0 and s[idx] >= threshold:
            r = refs[idx]
            err.append({"pair_id": r["pair_id"], "sample_id_a": r["a"], "sample_id_b": r["b"],
                        "problem_id_a": r["problem_a"], "problem_id_b": r["problem_b"],
                        "label": 0, "prediction": 1, "probability": float(s[idx]),
                        "error_type": "false_positive"})

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
                f.write(json.dumps({"pair_id": r["pair_id"], "label": int(t),
                                    "score": float(v)}) + "\n")
    _atomic_write_text(out_dir / "model_stats.json", json.dumps({
        "architecture": "Hybrid: GraphCodeBERT [CLS] + CPG-GNN [mean;max] -> "
                        "[hA; hB; |hA-hB|] -> MLP(Tanh) (DEC-005 concat_mlp, DEC-016 head)",
        "hf_id": cfg["model"]["hf_id"],
        "text_dim": model.text_dim,
        "graph_hidden_size": int(cfg["model"]["graph_hidden_size"]),
        "graph_num_layers": int(cfg["model"]["graph_num_layers"]),
        "graph_heads": int(cfg["model"]["graph_heads"]),
        "head_hidden_size": int(cfg["model"].get("head_hidden_size", 1024)),
        "edge_types": N_EDGE_TYPES,
        "param_count": model.param_count(),
        "trainable_param_count": model.trainable_param_count(),
        "max_length": int(cfg["tokenizer"]["max_length"]),
        "truncation": "text cut max_length (DEC-022); cpg max_nodes cap (xem CPGStore)",
    }, indent=2))
    eval_time = time.perf_counter() - t_start
    summary = {"threshold": threshold, "validation_f1_at_thr": thr["f1"],
               "test": results["test"], "eval_time_s": eval_time, "out_dir": str(out_dir)}
    _atomic_write_text(out_dir / "EVAL_COMPLETE.json",
                       json.dumps({"completed": True, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------- diagnostic
def diagnostic(cfg, device: torch.device) -> dict:
    import tempfile

    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])

    model = build_model(cfg, device)
    assert_device_setup(device, model)
    _warm_start(cfg, model, device)
    tokenizer = load_tokenizer(cfg["model"]["hf_id"])
    max_len = int(cfg["tokenizer"]["max_length"])

    stores = build_stores(cfg, pairs_dir, ["train"])
    store = stores["train"]
    refs = balanced_refs(filter_refs(store, load_pair_refs(pairs_dir / "train.jsonl"),
                                     "train"), 128)
    codes = _restrict_codes(load_code_map(processed, ["train"]), refs)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=tr["learning_rate"])
    _AMP["enabled"] = bool(tr.get("mixed_precision", False)) and device.type == "cuda"

    ta, ga, tb, gb, y, _ = collate_pair(tokenizer, codes, store,
                                        refs[: int(tr["batch_size"])], max_len, device)
    model.train()
    optimizer.zero_grad()
    logits = model(ta, ga, tb, gb)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    grad_norm = float(sum(p.grad.detach().float().norm().item() ** 2
                          for p in model.parameters() if p.grad is not None) ** 0.5)
    logit_std = float(logits.detach().float().std().item())
    assert logit_std > 0, "logit std = 0 — mo hinh chua hoc duoc gi, dung lai"
    assert grad_norm > 0, "grad = 0 — backward hong, dung lai"

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        res = train_model(cfg, Path(tmp), device, max_pairs=128, epochs=1,
                          time_budget_s=0, resume=False)
    wall = time.perf_counter() - t0
    assert res["history"][0]["train_loss"] < float("inf")
    report = {"logit_std": round(logit_std, 6), "grad_norm": round(grad_norm, 6),
              "loss_1step": round(float(loss.item()), 6),
              "param_count": model.param_count(),
              "trainable_param_count": model.trainable_param_count(),
              "pairs_per_second_diag": round(128 / max(wall, 1e-9), 2),
              "wall_s": round(wall, 1)}
    print(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------ preflight
def preflight(cfg, device: torch.device) -> dict:
    import shutil as _shutil

    assert N_EDGE_TYPES == 6, (
        f"schema doi (N_EDGE_TYPES={N_EDGE_TYPES}) — model embedding + gate "
        "notebook gia dinh 6 loai (DEC-027)")
    pre = cfg["preprocessing"]
    export_dir = Path(pre["export_dir"])
    cache_dir = Path(pre["cache_dir"])
    n_json = len(list(export_dir.glob("*.json"))) if export_dir.exists() else 0
    have_cache = [(cache_dir / f"cpp_{s}_full.npz").exists()
                  for s in ("train", "validation", "test")]
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    have_pairs = (pairs_dir.exists()
                  and all((pairs_dir / f"{s}.jsonl").exists()
                          for s in ("train", "validation", "test")))
    processed = Path(cfg["dataset"]["processed"])
    total, used, free = _shutil.disk_usage(".")
    vram = "n/a"
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        vram = f"{p.total_memory / 2**30:.1f}GB {p.name}"
    report = {
        "schema_edge_types": N_EDGE_TYPES,
        "export_jsons": n_json,
        "full_caches": have_cache,
        "pairs_built": have_pairs,
        "processed_exists": processed.exists(),
        "disk_free_gb": round(free / 2**30, 1),
        "vram": vram,
        "epochs": int(cfg["training"]["epochs"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "freeze_encoders": bool(cfg["training"].get("freeze_encoders", False)),
        "warm_start": cfg["model"].get("warm_start", {}),
    }
    print(json.dumps(report, indent=2))
    assert n_json > 50000, (
        f"export chi co {n_json} JSON — chua chay full extraction (can ~51.991)")
    assert all(have_cache), (
        "thieu cache full — chay notebook extraction cell 10 truoc")
    assert have_pairs, "chua co pairs — chay build_dataset.py truoc"
    assert processed.exists(), "thieu samples.jsonl — chay build_dataset.py truoc"
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["preflight", "smoke", "train", "evaluate",
                                       "diagnostic", "resume"], required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--time-budget-s", type=int, default=DEFAULT_TIME_BUDGET_S,
                    help="Dừng train gracefully sau N giây rồi chạy evaluate "
                         "(bảo vệ giới hạn phiên Kaggle 9h). 0 = tắt.")
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
        res = train_model(cfg, out_dir / "smoke", device, max_pairs=SMOKE_TRAIN_PAIRS,
                          epochs=1, time_budget_s=0)
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