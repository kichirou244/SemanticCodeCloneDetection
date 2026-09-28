"""EXP-004 — M3 ASTNN (Zhang et al., 2019) trên POJ-104 (C++).

Khác biệt với M1/M2 ở **data path**; mọi tầng còn lại (artifact bền vững, resume,
early stopping, chọn threshold trên validation, error analysis) cố tình dùng lại
nguyên vẹn từ `transformer_pair_runner` để so sánh RQ1/RQ2 không bị lệch do
implementation.

Điểm quan trọng về phương pháp: ASTNN **không truncate**. Toàn bộ cây statement
được giữ nên không chịu giới hạn `max_length` 256 như M1/M2 (vốn cắt mất 60,5%
token). Đổi lại, độ dài chuỗi phụ thuộc độ phức tạp của hàm chứ không phải độ dài
văn bản. Đây là lý do `diagnostic`/`preflight` báo cáo **số statement theo batch**
thay vì tỉ lệ truncation.
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
from src.data.access import stream_jsonl
from src.experiments.exp001_tfidf import (
    run_metadata, select_threshold, split_metrics, validate_dataset,
)
from src.experiments.transformer_pair_runner import (
    DEFAULT_TIME_BUDGET_S, _AMP, _assert_device_match, _atomic_write_text, _optimizer_step,
    _restrict_codes, assert_device_setup, collect_device_metadata, load_code_map, load_pair_refs,
    load_training_state, save_training_state, write_history,
)
from src.models.astnn.model import ASTNNPairClassifier, to_device
from src.representations.ast.store import ASTTreeStore

SMOKE_TRAIN_PAIRS = 256
SMOKE_VAL_PAIRS = 128
DIAG_TRAIN_PAIRS = 64
DIAG_VAL_PAIRS = 32

# Ngưỡng mở cảnh báo: số statement trung bình mỗi code. Vượt ngưỡng này thì một
# batch có thể vượt VRAM T4 16GB — phải giảm `batch_size` trong config.
WARN_STMT_PER_BATCH = 12000


# --------------------------------------------------------------------- store
def cache_root(cfg: dict, split: str, tag: str = "") -> Path:
    """Đường dẫn cache AST. Tách theo `max_code_chars` vì đổi ngưỡng này là đổi
    dữ liệu — dùng lại cache cũ sẽ train trên feature khác config.

    `tag` tách cache của smoke/diagnostic (chỉ vài trăm code) khỏi cache train đầy
    đủ. Bỏ `tag` này thì smoke sẽ ghi đè cache train và lần train sau sẽ âm thầm
    chạy trên vài trăm code thay vì 41k — loại lỗi im lặng, không có cảnh báo.
    """
    pre = cfg.get("preprocessing", {})
    cap = int(pre.get("max_code_chars", 20000))
    root = Path(pre.get("cache_dir", "artifacts/ast_cache"))
    suffix = f"_{tag}" if tag else ""
    return root / f"cpp_cap{cap}_{split}{suffix}.npz"


def build_stores(cfg: dict, processed: Path, splits: list[str],
                 restrict: dict[str, list[dict]] | None = None, tag: str = "",
                 verbose: bool = True) -> dict[str, ASTTreeStore]:
    """Compile cây AST cho từng split. `restrict` giới hạn codes cho smoke/diagnostic
    để không phải parse cả 41k codes khi chỉ forward vài chục pair."""
    lang = cfg["model"].get("language", "cpp")
    out: dict[str, ASTTreeStore] = {}
    for split in splits:
        codes = load_code_map(processed, [split])
        if restrict and split in restrict:
            codes = _restrict_codes(codes, restrict[split])
        out[split] = ASTTreeStore.build(codes, lang, cache_root(cfg, split, tag), verbose=verbose)
    return out


def ast_stats(store: ASTTreeStore) -> dict:
    """Thay `truncation_analysis`: ASTNN không truncate nên báo cáo phân bố cây."""
    s = store.stats
    return {
        "n_codes": len(store),
        "n_stmt_mean": s.get("n_stmt_mean"),
        "n_stmt_p95": s.get("n_stmt_p95"),
        "n_stmt_max": s.get("n_stmt_max"),
        "n_top_mean": s.get("n_top_mean"),
        "n_top_max": s.get("n_top_max"),
        "anc_pairs_mean": s.get("anc_pairs_mean"),
        "depth_max": s.get("depth_max"),
        "truncation": "none (ASTNN giữ toàn bộ cây statement)",
        "memory_note": s.get("memory_note"),
    }


# ------------------------------------------------------------------ batching
def collate_pair(store: ASTTreeStore, chunk: list[dict], device) -> tuple[dict, dict, torch.Tensor]:
    a = to_device(store.collate([r["a"] for r in chunk]), device)
    b = to_device(store.collate([r["b"] for r in chunk]), device)
    y = torch.tensor([float(r["label"]) for r in chunk], dtype=torch.float32, device=device)
    return a, b, y


def run_batches(model, store, refs, batch_size, device, optimizer=None, accum_steps: int = 1,
                max_grad_norm: float | None = None, scheduler=None, verbose: bool = True):
    """Một pass: train (optimizer khác None) hoặc eval.

    Trả về `loss, scores, labels, n_stmt, opt_steps` — `n_stmt` là tổng số statement
    đã xử lý (tương đương vị trí `n_trunc` của M1/M2 nhưng mang ý nghĩa khác).
    """
    train = optimizer is not None
    model.train(train)
    total_loss, n_stmt, opt_steps = 0.0, 0, 0
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
            for name, d in (("input_a", ba), ("input_b", bb)):
                for k, v in d.items():
                    if torch.is_tensor(v):
                        _assert_device_match(device, v, name=name)
            _assert_device_match(device, y, name="labels")
        n_stmt += int(ba["s_label"].shape[0])
        if train:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(ba, bb)
                if logits.device.type != device.type:
                    raise RuntimeError(
                        f"FAIL FAST: logits trên {logits.device}, kỳ vọng {device} (§7)")
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
            n_stmt, opt_steps)


def _fmt(sec: float) -> str:
    sec = int(sec)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"


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
    epochs = epochs or tr["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    model = ASTNNPairClassifier(
        int(cfg["model"]["hidden_size"]), float(tr.get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)  # FAIL FAST nếu CPU fallback (§7)

    train_refs = load_pair_refs(pairs_dir / "train.jsonl")
    val_refs = load_pair_refs(pairs_dir / "validation.jsonl")
    if max_pairs:  # smoke: subset cân bằng pos/neg, deterministic
        def balanced(refs, n):
            return ([r for r in refs if r["label"] == 1][: n // 2]
                    + [r for r in refs if r["label"] == 0][: n // 2])

        train_refs = balanced(train_refs, max_pairs)
        val_refs = balanced(val_refs, max_pairs // 2)

    # Chỉ parse codes mà smoke/diagnostic thực sự dùng; train đầy đủ thì cả split.
    restrict = None
    tag = ""
    if max_pairs:
        restrict = {"train": train_refs, "validation": val_refs}
        tag = f"smoke{max_pairs}"
    stores = build_stores(cfg, processed, ["train", "validation"], restrict, tag=tag)
    train_store, val_store = stores["train"], stores["validation"]

    stats_path = out_dir / "ast_stats.json"
    if resume and stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print("resume: dùng lại ast_stats.json")
    else:
        stats = {"train": ast_stats(train_store), "validation": ast_stats(val_store)}
        _atomic_write_text(stats_path, json.dumps(stats, indent=2))

    bs = int(tr["batch_size"])
    stmt_per_batch = stats["train"]["n_stmt_mean"] * 2 * bs
    if stmt_per_batch > WARN_STMT_PER_BATCH:
        print(f"WARNING: ~{stmt_per_batch:,.0f} statement/batch (ngưỡng "
              f"{WARN_STMT_PER_BATCH:,}); giảm batch_size nếu OOM trên T4")

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
        loss, _, _, n_stmt, opt_steps = run_batches(
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
                        "n_stmt_train": n_stmt,
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
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    meta = write_run_metadata(out_dir, cfg, manifest_path, device, model)
    meta.update({"training_time_s": train_time, "best_epoch": best_epoch,
                 "best_val_f1": best_f1, "total_optimizer_steps": global_step,
                 "stopped_reason": stopped_reason, "epochs_completed": len(history),
                 "time_budget_s": time_budget_s or None})
    _atomic_write_text(out_dir / "run_metadata.json", json.dumps(meta, indent=2))
    return {"best_epoch": best_epoch, "best_val_f1": best_f1, "history": history,
            "ast_stats": stats, "train_time_s": train_time,
            "stopped_reason": stopped_reason, "epochs_completed": len(history)}


# ------------------------------------------------------------------- evaluate
def evaluate_model(cfg, out_dir: Path, device: torch.device) -> dict:
    t_start = time.perf_counter()  # gồm cả inference, không chỉ thời gian ghi file
    manifest_path = Path(cfg["dataset"]["manifest"])
    pairs_dir = Path(cfg["dataset"]["pairs_dir"])
    processed = Path(cfg["dataset"]["processed"])
    validate_dataset(manifest_path, pairs_dir, processed)

    ckpt_path = out_dir / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Thiếu best checkpoint: {ckpt_path} (§23)")
    model = ASTNNPairClassifier(
        int(cfg["model"]["hidden_size"]), float(cfg["training"].get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)
    # weights_only=False: checkpoint chứa rng state (np.random.get_state) — torch>=2.6 chặn mặc định
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    model.eval()
    bs = int(cfg["inference"]["pair_batch_size"])

    stores = build_stores(cfg, processed, ["validation", "test"])
    scored = {}
    for split in ("validation", "test"):
        refs = load_pair_refs(pairs_dir / f"{split}.jsonl")
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
        "architecture": "ASTNN: RvNN trên cây statement + Bi-GRU trên statement cấp 1 "
                        "+ [hA; hB; |hA-hB|] + MLP(Tanh) — DEC-016",
        "hidden_size": int(cfg["model"]["hidden_size"]),
        "param_count": model.param_count(),
        "language": cfg["model"].get("language", "cpp"),
        "truncation": "none",
    }, indent=2))
    eval_time = time.perf_counter() - t_start
    summary = {"threshold": threshold, "validation_f1_at_thr": thr["f1"],
               "test": results["test"], "eval_time_s": eval_time, "out_dir": str(out_dir)}
    _atomic_write_text(out_dir / "EVAL_COMPLETE.json",
                       json.dumps({"completed": True, "summary": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ------------------------------------------------------------------- diagnostic
def diagnostic(cfg, device: torch.device) -> dict:
    """Verify device/gradients/param-update/logit variance + **đo throughput thật**.

    Phần đo throughput là thứ quyết định ước lượng thời gian train: ta chạy N
    optimizer step trên batch *thật* (không phải batch nhỏ), rồi ngoại suy ra
    `sec/step` × tổng số step. Ước lượng chỉ đáng tin trên **đúng GPU đích** —
    chạy lại `diagnostic` trên Kaggle T4 trước khi báo cáo (§22, §33).
    """
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
    bs = int(tr["batch_size"])

    model = ASTNNPairClassifier(
        int(cfg["model"]["hidden_size"]), float(tr.get("dropout", 0.2))).to(device)
    assert_device_setup(device, model)
    mdev = str(next(model.parameters()).device)
    print(json.dumps({"Model device": mdev}))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_train = sum(p.numel() for p in model.rvnn.parameters() if p.requires_grad)
    gru_train = sum(p.numel() for p in model.gru.parameters() if p.requires_grad)
    head_train = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    print(json.dumps({"trainable_params": trainable, "rvnn_trainable": emb_train,
                      "gru_trainable": gru_train, "classifier_trainable": head_train}))
    assert emb_train > 0 and gru_train > 0 and head_train > 0, \
        "FAIL: thành phần nào của ASTNN bị freeze toàn bộ"

    refs = load_pair_refs(pairs_dir / "train.jsonl")
    # Mẫu ngẫu nhiên có seed, phủ đều cả pos/neg và cả đuôi file. Lấy "N cái đầu
    # tiên" sẽ thiên lệch về phần code đơn giản của dataset → sec/step thấp hơn
    # thực tế → ước lượng thời gian lạc quan.
    seed = tr["seed"]
    bench_pool = random.Random(f"astnn-bench/{seed}").sample(
        refs, min(2 * bs, len(refs)))
    pos = [r for r in bench_pool if r["label"] == 1][:bs]
    neg = [r for r in bench_pool if r["label"] == 0][:bs]
    bench_refs = (pos + neg)[:bs]
    assert len(pos) and len(neg), "FAIL: diagnostic subset thiếu pos/neg"
    store = build_stores(cfg, processed, ["train"], {"train": bench_refs},
                         tag=f"diagbench{bs}")["train"]

    # Pool riêng cho correctness check: benchmark trên chính các pair nó vừa train
    # 21 step sẽ ghi nhớ, lúc đó `logits_vary`/`probs_not_constant` trở nên vô nghĩa
    # (mọi prob dồn về 0/1). Và pool check phải có codes riêng để không vướng
    # `ASTTreeStore.index` của pool bench.
    check_refs = random.Random(f"astnn-check/{seed}").sample(
        refs, min(DIAG_TRAIN_PAIRS, len(refs)))
    store_check = build_stores(cfg, processed, ["train"], {"train": check_refs},
                               tag=f"diagcheck{bs}")["train"]

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = AdamW(model.parameters(), lr=tr["learning_rate"])
    _AMP["enabled"] = False  # diagnostic: tắt AMP cho rõ ràng (§29)

    # --- warmup: loại khỏi phép đo (allocator + cuDNN autotune) -------------
    run_batches(model, store, bench_refs, bs, device, optimizer, accum_steps=1,
                max_grad_norm=tr.get("max_grad_norm"), verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # --- đo throughput trên batch thật -------------------------------------
    n_bench = int(cfg.get("diagnostic", {}).get("bench_steps", 20))
    n_micro_per_step = max(1, int(tr["gradient_accumulation_steps"]))
    t0 = time.perf_counter()
    for _ in range(n_bench):
        run_batches(model, store, bench_refs, bs, device, optimizer,
                    accum_steps=n_micro_per_step, max_grad_norm=tr.get("max_grad_norm"),
                    verbose=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / n_bench

    n_train = sum(1 for _ in stream_jsonl(pairs_dir / "train.jsonl"))
    epochs = int(tr["epochs"])
    steps = math.ceil(n_train / bs / tr["gradient_accumulation_steps"]) * epochs
    train_s = per_step * steps
    per_epoch = math.ceil(n_train / bs / tr["gradient_accumulation_steps"])
    val_refs_n = sum(1 for _ in stream_jsonl(pairs_dir / "validation.jsonl"))
    val_steps = math.ceil(val_refs_n / (bs * 2))
    val_s = per_step * 0.4 * val_steps * epochs  # forward-only ≈ 40% cost/step

    timing = {
        "bench_steps": n_bench,
        "batch_size": bs,
        "gradient_accumulation_steps": n_micro_per_step,
        "sec_per_optimizer_step": round(per_step, 4),
        "steps_per_epoch": per_epoch,
        "total_train_optimizer_steps": steps,
        "projected_train_time_s": round(train_s, 1),
        "projected_train_time_h": round(train_s / 3600, 2),
        "projected_val_time_s_all_epochs": round(val_s, 1),
        "projected_total_s": round(train_s + val_s, 1),
        "n_stmt_per_batch": int(store.stats.get("n_stmt_mean", 0)) * 2 * bs,
    }
    print(json.dumps({"THROUGHPUT": timing}, indent=2))
    print(f"==> dự kiến train {timing['projected_train_time_h']}h "
          f"+ val {timing['projected_val_time_s_all_epochs'] / 3600:.2f}h "
          f"= {timing['projected_total_s'] / 3600:.2f}h trên {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # --- correctness checks -------------------------------------------------
    losses, s, y = [], None, None
    for epoch in range(2):
        loss, s, y, _, _ = run_batches(model, store_check, check_refs, bs, device,
                                       optimizer, accum_steps=1,
                                       max_grad_norm=tr.get("max_grad_norm"))
        losses.append(loss)
        print(json.dumps({"epoch": epoch, "loss": loss,
                          "labels_unique": sorted(set(int(v) for v in y)),
                          "score_min": float(s.min()), "score_max": float(s.max()),
                          "score_mean": float(s.mean()), "score_std": float(s.std()),
                          "input_device": "cuda" if device.type == "cuda" else "cpu"}))

    model.train(True)
    optimizer.zero_grad()
    ba, bb, yb = collate_pair(store_check, check_refs[:bs], device)
    _loss = nn.BCEWithLogitsLoss()(model(ba, bb), yb)
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
    checks = {
        "cuda_confirmed": device.type != "cuda" or torch.cuda.is_available(),
        "model_on_cuda": mdev.startswith("cuda") if device.type == "cuda" else True,
        "rvnn_trained": emb_train > 0,
        "gru_trained": gru_train > 0,
        "gradients_finite": grads_ok and grad_norm > 0,
        "params_updated": changed > 0,
        "logits_vary": float(s.std()) > 0,
        "probs_not_constant": not (float(s.min()) == float(s.max()) == 0.5),
        "loss_finite": all(np.isfinite(losses)),
    }
    checks["grad_norm"] = grad_norm
    checks["params_changed"] = changed
    checks["losses"] = losses
    print(json.dumps({"DIAGNOSTIC": "PASS" if all(v for k, v in checks.items()
                    if isinstance(v, bool)) else "FAIL",
                      **{k: v for k, v in checks.items() if isinstance(v, bool)}}))
    if torch.cuda.is_available():
        print(json.dumps({"gpu_memory_allocated": torch.cuda.memory_allocated(0),
                          "gpu_memory_reserved": torch.cuda.memory_reserved(0)}))
    return {**checks, "throughput": timing}


# ------------------------------------------------------------------- preflight
def preflight(cfg, device: torch.device) -> dict:
    import transformers

    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,  # chỉ để đối chiếu môi trường
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
    model = ASTNNPairClassifier(int(cfg["model"]["hidden_size"]),
                                float(cfg["training"].get("dropout", 0.2)))
    info.update({
        "architecture": "ASTNN",
        "language": cfg["model"].get("language", "cpp"),
        "hidden_size": int(cfg["model"]["hidden_size"]),
        "param_count": model.param_count(),
        "batch_size": int(cfg["training"]["batch_size"]),
        "gradient_accumulation_steps": int(cfg["training"]["gradient_accumulation_steps"]),
        "mixed_precision": bool(cfg["training"].get("mixed_precision", False)),
        "max_length": "n/a (không truncate)",
        "threshold_method": cfg["threshold"]["method"],
    })
    info["n_train_pairs"] = sum(
        1 for _ in stream_jsonl(Path(cfg["dataset"]["pairs_dir"]) / "train.jsonl"))
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
