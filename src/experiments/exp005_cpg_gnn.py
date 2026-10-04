"""EXP-005 — M4 CPG-GNN: train + evaluate trên CPG đã trích (POJ-104, C++).

Chạy trên Kaggle GPU, SAU notebook extraction (bước 1):
    python -m src.experiments.exp005_cpg_gnn --config configs/exp005_cpg_gnn.yaml \
        --mode preflight --device cuda
    python -m src.experiments.exp005_cpg_gnn --config ... --mode smoke ...
    python -m src.experiments.exp005_cpg_gnn --config ... --mode train ...
    # train tự evaluate khi xong; --mode evaluate chạy riêng khi cần.

Dữ liệu vào: `artifacts/cpg_export/*.json` (Joern) + `data/pairs/poj104/*.jsonl`.
Hợp đồng artifacts RA giống hệt M1/M2/M3 (DEC-016 head, threshold trên
validation, test một lần): threshold.json, metrics.json/csv,
confusion_matrix.json, error_cases.jsonl, predictions/*_scores.jsonl,
model_stats.json, EVAL_COMPLETE.json, training_history.csv.

Khác biệt với M1/M2/M3 ở **data path** (đồ thị CPG qua CPGStore thay vì tensor
text); mọi tầng còn lại (artifact bền vững, resume, early stopping, threshold)
giữ nguyên để phép so sánh công bằng ở mức protocol.
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
from src.experiments.transformer_pair_runner import (
    DEFAULT_TIME_BUDGET_S, _AMP, _assert_device_match, _atomic_write_text,
    _optimizer_step, assert_device_setup, collect_device_metadata,
    load_eval_weights, load_pair_refs, load_training_state,
    resolve_eval_checkpoint, save_training_state, write_history,
)
from src.models.cpg_gnn.model import CPGGNNPairClassifier, to_device
from src.representations.cpg.graph import collate_pyg
from src.representations.cpg.schema import N_EDGE_TYPES
from src.representations.cpg.store import CPGStore

SMOKE_TRAIN_PAIRS = 64
# Tỉ lệ cặp thiếu đồ thị tối đa: extraction full đo 0 missing trên 51.991 hàm,
# nên vượt ngưỡng này là exporter đổi hành vi chứ không phải nhiễu.
MAX_MISSING_PAIR_RATE = 0.05


def required_ids(refs: list[dict]) -> list[str]:
    """Mọi sample_id mà split này cần — chỉ nạp đúng phần đó, không nạp thừa."""
    return sorted({r["a"] for r in refs} | {r["b"] for r in refs})


def build_stores(cfg: dict, pairs_dir: Path, splits: list[str]) -> dict[str, CPGStore]:
    """Một CPGStore cho mỗi split, dùng cache full của notebook extraction."""
    pre = cfg["preprocessing"]
    export_dir = Path(pre["export_dir"])
    cache_dir = Path(pre["cache_dir"])
    max_nodes = int(pre["max_nodes"])
    stores = {}
    for split in splits:
        refs = load_pair_refs(pairs_dir / f"{split}.jsonl")
        ids = required_ids(refs)
        store = CPGStore.build(export_dir, ids, max_nodes,
                               cache_dir / f"cpp_{split}_full.npz")
        stores[split] = store
        print(json.dumps({split: {"need_ids": len(ids), "have": len(store),
                                  "coverage": round(len(store) / max(len(ids), 1), 4)}}))
    return stores


def filter_refs(store: CPGStore, refs: list[dict], split: str) -> list[dict]:
    """Bỏ cặp thiếu đồ thị (một trong hai phía không có JSON). Fail-fast nếu
    tỉ lệ vượt ngưỡng — im lặng bỏ nhiều là train trên phân phối lệch."""
    kept = [r for r in refs if store.has(r["a"]) and store.has(r["b"])]
    skipped = len(refs) - len(kept)
    rate = skipped / max(len(refs), 1)
    print(json.dumps({f"{split}_pairs": len(refs), "kept": len(kept),
                      "skipped_no_graph": skipped, "skip_rate": round(rate, 4)}))
    assert rate <= MAX_MISSING_PAIR_RATE, (
        f"{split}: thieu do thi {skipped}/{len(refs)} ({rate:.1%}) > 5% — "
        "exporter doi hanh vi, xem lai extraction truoc khi train")
    return kept


def collate_pair(store: CPGStore, chunk: list[dict], device):
    a = to_device(store.collate([r["a"] for r in chunk]), device)
    b = to_device(store.collate([r["b"] for r in chunk]), device)
    y = torch.tensor([float(r["label"]) for r in chunk], dtype=torch.float32, device=device)
    return a, b, y


def _assert_batch_on_device(device, batch, name: str) -> None:
    for key in ("node_type", "x", "edge_index", "edge_type"):
        _assert_device_match(device, getattr(batch, key), name=f"{name}.{key}")


def run_batches(model, store, refs, batch_size, device, optimizer=None, accum_steps: int = 1,
                max_grad_norm: float | None = None, scheduler=None, verbose: bool = True):
    """Một pass: train (optimizer khác None) hoặc eval.

    Trả về `loss, scores, labels, n_nodes, opt_steps` — `n_nodes` để đo
    throughput (tương đương vị trí `n_stmt` của ASTNN).
    """
    train = optimizer is not None
    model.train(train)
    total_loss, n_nodes, opt_steps = 0.0, 0, 0
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
        ba, bb, y = collate_pair(store, chunk, device)
        if device.type == "cuda":
            _assert_batch_on_device(device, ba, "input_a")
            _assert_batch_on_device(device, bb, "input_b")
            _assert_device_match(device, y, name="labels")
        n_nodes += int(ba.node_type.shape[0]) + int(bb.node_type.shape[0])
        if train:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ba, bb)
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
                logits = model(ba, bb)
                loss = criterion(logits, y)
            total_loss += loss.item() * len(chunk)
        seen += len(chunk)
        all_s.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_y.append(np.array([r["label"] for r in chunk]))
        if verbose and ((ci + 1) % log_every == 0 or (ci + 1) == total_chunks):
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
            n_nodes, opt_steps)


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


# ---------------------------------------------------------------------- train
def train_model(cfg, out_dir: Path, device: torch.device, max_pairs: int = 0, epochs: int = 0,
                time_budget_s: int = 0, resume: bool = False) -> dict:
    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    mo = cfg["model"]
    epochs = epochs or tr["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    validate_dataset(manifest_path, pairs_dir, Path(cfg["dataset"]["processed"]))

    model = CPGGNNPairClassifier(
        int(mo["hidden_size"]), int(mo["num_layers"]), int(mo["heads"]),
        float(mo.get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)  # FAIL FAST nếu CPU fallback
    print(json.dumps({"param_count": model.param_count(),
                      "edge_types": N_EDGE_TYPES}))

    train_refs = load_pair_refs(pairs_dir / "train.jsonl")
    val_refs = load_pair_refs(pairs_dir / "validation.jsonl")
    if max_pairs:  # smoke: subset cân bằng pos/neg, deterministic
        def balanced(refs, n):
            return ([r for r in refs if r["label"] == 1][: n // 2]
                    + [r for r in refs if r["label"] == 0][: n // 2])

        train_refs = balanced(train_refs, max_pairs)
        val_refs = balanced(val_refs, max_pairs // 2)

    stores = build_stores(cfg, pairs_dir, ["train", "validation"])
    train_refs = filter_refs(stores["train"], train_refs, "train")
    val_refs = filter_refs(stores["validation"], val_refs, "validation")
    train_store, val_store = stores["train"], stores["validation"]

    bs = int(tr["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_refs) / bs / tr["gradient_accumulation_steps"]))
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"],
                      weight_decay=tr.get("weight_decay", 0.0))
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
        if start_epoch >= epochs:
            print(json.dumps({"WARNING": "đã train đủ epoch, chỉ cần evaluate"}))

    t0 = time.perf_counter()
    prev_elapsed = 0.0
    stopped_reason = "epochs_exhausted"
    for epoch in range(start_epoch, epochs):
        random.Random(f"{tr['seed']}/{epoch}").shuffle(train_refs)
        loss, _, _, n_nodes, opt_steps = run_batches(
            model, train_store, train_refs, bs, device, optimizer,
            accum_steps=tr["gradient_accumulation_steps"],
            max_grad_norm=tr.get("max_grad_norm"), scheduler=scheduler)
        global_step += opt_steps
        v_loss, v_s, v_y, _, _ = run_batches(
            model, val_store, val_refs, bs * 2, device)
        vm = split_metrics(v_s, v_y, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": v_loss,
                        "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "lr": lr,
                        "n_nodes_train": n_nodes,
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
    meta = write_run_metadata(out_dir, cfg, manifest_path, device, model)
    meta.update({"training_time_s": train_time, "best_epoch": best_epoch,
                 "best_val_f1": best_f1, "total_optimizer_steps": global_step,
                 "stopped_reason": stopped_reason, "epochs_completed": len(history),
                 "time_budget_s": time_budget_s or None})
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history,
            "train_time_s": train_time,
            "stopped_reason": stopped_reason, "epochs_completed": len(history)}


# ------------------------------------------------------------------- evaluate
def evaluate_model(cfg, out_dir: Path, device: torch.device) -> dict:
    t_start = time.perf_counter()  # gồm cả inference, không chỉ thời gian ghi file
    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    validate_dataset(manifest_path, pairs_dir, Path(cfg["dataset"]["processed"]))

    ckpt_path = resolve_eval_checkpoint(out_dir)
    mo = cfg["model"]
    model = CPGGNNPairClassifier(
        int(mo["hidden_size"]), int(mo["num_layers"]), int(mo["heads"]),
        float(mo.get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)
    model.load_state_dict(load_eval_weights(ckpt_path, device))
    model.eval()
    bs = int(cfg["inference"]["pair_batch_size"])

    stores = build_stores(cfg, pairs_dir, ["validation", "test"])
    scored = {}
    for split in ("validation", "test"):
        refs = filter_refs(stores[split], load_pair_refs(pairs_dir / f"{split}.jsonl"), split)
        _, s, y, _, _ = run_batches(model, stores[split], refs, bs, device)
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
        "architecture": "CPG-GNN: node emb + TypedGAT(3 tầng, 4 heads, edge-conditioned) "
                        "+ residual + [mean;max] + [hA; hB; |hA-hB|] + MLP(Tanh) — DEC-016",
        "hidden_size": int(mo["hidden_size"]),
        "num_layers": int(mo["num_layers"]),
        "heads": int(mo["heads"]),
        "edge_types": N_EDGE_TYPES,
        "param_count": model.param_count(),
        "language": cfg["model"].get("language", "cpp"),
        "truncation": "none (max_nodes cap, xem CPGStore)",
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
    """Verify forward/backward/grad/logit-std + đo throughput thật trên GPU.

    Nạp store đầy đủ (đường dữ liệu thật), 1 forward+backward trên 1 batch,
    rồi 1 epoch trên 128 cặp vào thư mục tạm. Phải in `logit std > 0`;
    nếu scores hội tụ về 0 thì chưa học được, dừng lại.
    """
    import tempfile

    set_global_seed(cfg["training"]["seed"])
    tr = cfg["training"]
    mo = cfg["model"]
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    stores = build_stores(cfg, pairs_dir, ["train"])
    store = stores["train"]

    def balanced(refs, n):
        return ([r for r in refs if r["label"] == 1][: n // 2]
                + [r for r in refs if r["label"] == 0][: n // 2])

    refs = balanced(filter_refs(store, load_pair_refs(pairs_dir / "train.jsonl"),
                                "train"), 128)
    model = CPGGNNPairClassifier(
        int(mo["hidden_size"]), int(mo["num_layers"]), int(mo["heads"]),
        float(mo.get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])
    _AMP["enabled"] = bool(tr.get("mixed_precision", False)) and device.type == "cuda"

    # 1 forward + backward riêng để lấy grad/logit-std (không lẫn vào train).
    ba, bb, y = collate_pair(store, refs[: int(tr["batch_size"])], device)
    model.train()
    optimizer.zero_grad()
    logits = model(ba, bb)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    grad_norm = float(sum(p.grad.detach().float().norm().item() ** 2
                          for p in model.parameters() if p.grad is not None) ** 0.5)
    logit_std = float(logits.detach().float().std().item())
    assert logit_std > 0, "logit std = 0 — mo hinh chua hoc duoc gi, dung lai"
    assert grad_norm > 0, "grad = 0 — backward hong, dung lai"

    # 1 epoch ngắn để xác nhận cả vòng train (scheduler, checkpoint, history).
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        res = train_model(cfg, Path(tmp), device, max_pairs=128, epochs=1,
                          time_budget_s=0, resume=False)
    wall = time.perf_counter() - t0
    assert res["history"][0]["train_loss"] < float("inf")
    report = {"logit_std": round(logit_std, 6), "grad_norm": round(grad_norm, 6),
              "loss_1step": round(float(loss.item()), 6),
              "param_count": model.param_count(),
              "pairs_per_second_diag": round(128 / max(wall, 1e-9), 2),
              "wall_s": round(wall, 1)}
    print(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------ preflight
def preflight(cfg, device: torch.device) -> dict:
    """Gate trước khi tốn giờ GPU: config + schema + dữ liệu extraction + VRAM."""
    import shutil as _shutil

    assert N_EDGE_TYPES == 6, (
        f"schema doi (N_EDGE_TYPES={N_EDGE_TYPES}) — model embedding + gate "
        "notebook gia dinh 6 loai (DEC-027)")
    pre = cfg["preprocessing"]
    export_dir = Path(pre["export_dir"])
    cache_dir = Path(pre["cache_dir"])
    n_json = len(list(export_dir.glob("*.json"))) if export_dir.exists() else 0
    have_cache = [ (cache_dir / f"cpp_{s}_full.npz").exists()
                   for s in ("train", "validation", "test") ]
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    have_pairs = (pairs_dir.exists()
                  and all((pairs_dir / f"{s}.jsonl").exists()
                          for s in ("train", "validation", "test")))
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
        "disk_free_gb": round(free / 2**30, 1),
        "vram": vram,
        "epochs": int(cfg["training"]["epochs"]),
        "batch_size": int(cfg["training"]["batch_size"]),
    }
    print(json.dumps(report, indent=2))
    assert n_json > 50000, (
        f"export chi co {n_json} JSON — chua chay full extraction (can ~51.991)")
    assert all(have_cache), (
        "thieu cache full — chay notebook extraction cell 10 truoc")
    assert have_pairs, "chua co pairs — chay build_dataset.py truoc"
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
