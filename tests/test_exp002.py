"""Tests EXP-002 (CPU): model, tokenizer, checkpoint, threshold, leakage, runner (§36).

Chạy trên máy user (cần checkpoint microsoft/codebert-base trong HF cache —
đã có từ smoke_transformer ở INIT).
"""

import json

import pytest
import torch

from src.experiments.exp002_codebert import (
    evaluate_model,
    export_eval_weights,
    load_code_map,
    load_eval_weights,
    load_pair_refs,
    preflight,
    resolve_eval_checkpoint,
    train_model,
    truncation_analysis,
)
from src.experiments.transformer_pair_runner import EVAL_WEIGHTS_NAME
from src.experiments.exp001_tfidf import validate_dataset
from src.models.codebert_pair import (
    CodeBERTPairClassifier,
    collate_pair_batch,
    load_tokenizer,
)

HF_ID = "microsoft/codebert-base"


def test_pretrained_loads_with_correct_dims():
    m = CodeBERTPairClassifier(HF_ID)
    assert m.hidden_size == 768
    assert m.param_count() > 100_000_000  # ~125M + head, không phải random-tiny


def test_forward_backward_finite():
    torch.manual_seed(0)
    m = CodeBERTPairClassifier(HF_ID)
    tok = load_tokenizer(HF_ID)
    device = torch.device("cpu")
    m.to(device)
    ba, bb, y, _ = collate_pair_batch(
        tok, ["int add(int a,int b){return a+b;}", "void f(){}"],
        ["int sum(int x,int y){return x+y;}", "void f(){}"],
        [1, 0], 32, device,
    )
    logits = m(ba, bb)
    assert logits.shape == (2,)
    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_tokenizer_deterministic_and_truncation_tracked():
    tok = load_tokenizer(HF_ID)
    device = torch.device("cpu")
    codes = ["int main(){" + "a=1;" * 200 + "}"]
    b1, _, _, t1 = collate_pair_batch(tok, codes, codes, [1], 32, device)
    b2, _, _, t2 = collate_pair_batch(tok, codes, codes, [1], 32, device)
    assert t1 == 2 and t2 == 2  # cả 2 phía đều bị truncate
    assert torch.equal(b1["input_ids"], b2["input_ids"])  # deterministic
    assert b1["input_ids"].shape[1] <= 32  # tôn trọng max_length


def test_checkpoint_save_load(tmp_path):
    m = CodeBERTPairClassifier(HF_ID)
    p = tmp_path / "ckpt.pt"
    torch.save({"model_state": m.state_dict(), "epoch": 0}, p)
    m2 = CodeBERTPairClassifier(HF_ID)
    m2.load_state_dict(torch.load(p, map_location="cpu")["model_state"])
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)


def test_truncation_analysis_counts():
    tok = load_tokenizer(HF_ID)
    codes = {"s": "int x=1;", "l": "int y=" + "2+" * 500 + "3;"}
    r = truncation_analysis(tok, codes, 32)
    assert r == {"n": 2, "n_truncated": 1, "rate": 0.5}


def _tiny_dataset(tmp_path):
    proc = tmp_path / "samples.jsonl"
    rows = [
        {"sample_id": "a0", "code": "int add(int x,int y){return x+y;}", "problem_id": "p1", "split": "validation"},
        {"sample_id": "a1", "code": "int sum(int x,int y){return x+y;}", "problem_id": "p1", "split": "validation"},
        {"sample_id": "b0", "code": "void print(){printf(\"hi\");}", "problem_id": "p2", "split": "validation"},
        {"sample_id": "t0", "code": "int add(int x,int y){return x+y;}", "problem_id": "p3", "split": "test"},
        {"sample_id": "t1", "code": "void print(){printf(\"hi\");}", "problem_id": "p3", "split": "test"},
    ]
    with open(proc, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "train.jsonl").write_text("", encoding="utf-8")  # validate cần đủ 3 files
    for split, ps in [("validation", [("v0", "a0", "a1", 1), ("v1", "a0", "b0", 0)]),
                      ("test", [("e0", "t0", "t0", 1), ("e1", "t0", "t1", 0)])]:
        with open(pairs_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for pid, a, b, lab in ps:
                f.write(json.dumps({"pair_id": pid, "code_a_id": a, "code_b_id": b,
                                    "problem_a": "px", "problem_b": "py",
                                    "label": lab, "split": split}) + "\n")
    manifest = {"dataset": "tiny", "leakage_passed": True}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return proc, pairs_dir, mp


def test_dataset_refs_no_split_mixing(tmp_path):
    proc, pairs_dir, _ = _tiny_dataset(tmp_path)
    refs = load_pair_refs(pairs_dir / "validation.jsonl")
    assert all(r["label"] in (0, 1) for r in refs) and len(refs) == 2
    codes = load_code_map(proc, ["validation"])
    assert set(codes) == {"a0", "a1", "b0"}  # không lẫn test


def test_manifest_fail_fast(tmp_path):
    import pytest

    bad = tmp_path / "m.json"
    bad.write_text(json.dumps({"leakage_passed": False}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        validate_dataset(bad, tmp_path, tmp_path / "x.jsonl")


def test_evaluate_untrained_model_runs(tmp_path):
    """Smoke evaluate với checkpoint khởi tạo (weights random từ head mới chỉ
    để verify pipeline artifact; KHÔNG phải paper result)."""
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    m = CodeBERTPairClassifier(HF_ID)
    torch.save({"model_state": m.state_dict()}, out / "best_checkpoint.pt")
    cfg = {
        "dataset": {"manifest": str(mp), "pairs_dir": str(pairs_dir), "processed": str(proc)},
        "model": {"hf_id": HF_ID},
        "tokenizer": {"max_length": 32},
        "inference": {"pair_batch_size": 2},
        "threshold": {"grid_step": 0.1},
        "error_analysis": {"top_k": 10},
    }
    summary = evaluate_model(cfg, out, torch.device("cpu"))
    for name in ("metrics.json", "metrics.csv", "threshold.json",
                 "confusion_matrix.json", "error_cases.jsonl", "model_stats.json"):
        assert (out / name).exists(), name
    assert (out / "predictions" / "test_scores.jsonl").exists()
    assert set(summary["test"]) >= {"precision", "recall", "f1"}


def test_preflight_cuda_failfast_or_reports(tmp_path):
    import pytest

    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "train.jsonl").write_text('{"pair_id": "x"}\n', encoding="utf-8")
    cfg = {
        "model": {"hf_id": HF_ID},
        "tokenizer": {"max_length": 256},
        "training": {"batch_size": 16, "gradient_accumulation_steps": 2},
        "threshold": {"method": "validation_f1"},
        "dataset": {"pairs_dir": str(pairs_dir)},
    }
    if torch.cuda.is_available():
        info = preflight(cfg, torch.device("cuda"))
        assert "gpu_name" in info and info["param_count"] > 0
    else:
        with pytest.raises(RuntimeError):  # máy test không CUDA → fail-fast
            preflight(cfg, torch.device("cuda"))


# --------------------------------------------------------------- Kaggle resilience (DEC-018)
class _TinyPairModel(torch.nn.Module):
    """Model siêu nhỏ thay CodeBERT cho test orchestration (resume / budget / artifact).

    KHÔNG dùng để đo chất lượng — chỉ để test logic resume + flush. Giữ đúng
    interface mà `train_model` cần: hidden_size, encoder, head, forward, param_count.
    Nhờ vậy test không tải 126M params / không gọi mạng / không chậm.
    """

    def __init__(self, hf_id: str, dropout: float = 0.1):
        super().__init__()
        self.hf_id = hf_id
        self.hidden_size = 8
        self.encoder = torch.nn.Module()
        self.encoder.emb = torch.nn.Embedding(50265, self.hidden_size)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * self.hidden_size, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
        )

    def encode_side(self, inputs: dict) -> torch.Tensor:
        return self.encoder.emb(inputs["input_ids"]).mean(dim=1)

    def forward(self, inputs_a: dict, inputs_b: dict) -> torch.Tensor:
        h_a, h_b = self.encode_side(inputs_a), self.encode_side(inputs_b)
        pair = torch.cat([h_a, h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.head(pair).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


@pytest.fixture
def tiny_model(monkeypatch):
    # Patch vào `transformer_pair_runner` — module thật sự chứa code train. exp002/exp003
    # chỉ là wrapper re-export, nên patch vào chúng sẽ không có tác dụng.
    import src.experiments.transformer_pair_runner as runner

    monkeypatch.setattr(runner, "TransformerPairClassifier", _TinyPairModel)
    return runner


def _tiny_train_dataset(tmp_path):
    """Dataset nhỏ có train/val/test để chạy train_model end-to-end."""
    proc = tmp_path / "samples.jsonl"
    code = "int add(int x,int y){return x+y;}"
    rows = []
    for i in range(8):
        rows.append({"sample_id": f"r{i}", "code": code, "problem_id": "p1", "split": "train"})
    for i in range(4):
        rows.append({"sample_id": f"v{i}", "code": code, "problem_id": "p2", "split": "validation"})
    for i in range(4):
        rows.append({"sample_id": f"t{i}", "code": code, "problem_id": "p3", "split": "test"})
    with open(proc, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    for split, prefix in [("train", "r"), ("validation", "v"), ("test", "t")]:
        with open(pairs_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"pair_id": f"{prefix}0", "code_a_id": f"{prefix}0",
                                "code_b_id": f"{prefix}1", "problem_a": "px",
                                "problem_b": "px", "label": 1, "split": split}) + "\n")
            f.write(json.dumps({"pair_id": f"{prefix}n", "code_a_id": f"{prefix}0",
                                "code_b_id": f"{prefix}2", "problem_a": "px",
                                "problem_b": "py", "label": 0, "split": split}) + "\n")
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps({"dataset": "tiny", "leakage_passed": True}), encoding="utf-8")
    return proc, pairs_dir, mp


def _train_cfg(tmp_path, proc, pairs_dir, mp, epochs=2):
    return {
        "_config_path": str(tmp_path / "cfg.yaml"),
        "dataset": {"manifest": str(mp), "pairs_dir": str(pairs_dir), "processed": str(proc)},
        "model": {"hf_id": HF_ID, "dropout": 0.1},
        "tokenizer": {"max_length": 32},
        "training": {"seed": 42, "epochs": epochs, "batch_size": 2, "learning_rate": 2e-5,
                     "weight_decay": 0.01, "warmup_ratio": 0.1,
                     "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                     "mixed_precision": False},
        "early_stopping": {"enabled": False, "patience": 2},
    }


def test_artifacts_flushed_every_epoch(tmp_path, tiny_model):
    """DEC-018: history/metadata/checkpoint phải có sau TỪNG epoch, không đợi hết run."""
    proc, pairs_dir, mp = _tiny_train_dataset(tmp_path)
    cfg = _train_cfg(tmp_path, proc, pairs_dir, mp, epochs=1)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    train_model(cfg, out, torch.device("cpu"))
    for name in ("training_history.csv", "run_metadata.json", "truncation.json",
                 "last_checkpoint.pt", "best_checkpoint.pt"):
        assert (out / name).exists(), name
    # Không còn checkpoint từng epoch (tiết kiệm ~3GB trên Kaggle)
    assert not list(out.glob("checkpoint_epoch*.pt"))
    rows = (out / "training_history.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2  # header + 1 epoch
    assert "epoch_time_s" in rows[0] and "global_step" in rows[0]


def test_resume_continues_instead_of_restarting(tmp_path, tiny_model):
    """Session chết giữa chừng → train tiếp epoch còn lại, không chạy lại từ đầu."""
    proc, pairs_dir, mp = _tiny_train_dataset(tmp_path)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    cfg = _train_cfg(tmp_path, proc, pairs_dir, mp, epochs=2)
    first = train_model(cfg, out, torch.device("cpu"), epochs=1)
    assert first["epochs_completed"] == 1

    second = train_model(cfg, out, torch.device("cpu"), epochs=2, resume=True)
    assert second["history"][0]["epoch"] == "0" or second["history"][0]["epoch"] == 0
    epochs_seen = [int(r["epoch"]) for r in second["history"]]
    assert epochs_seen == [0, 1], epochs_seen  # nối tiếp, không mất epoch 0
    assert second["history"][-1]["global_step"] > first["history"][-1]["global_step"]


def test_resume_without_checkpoint_fails_fast(tmp_path, tiny_model):
    proc, pairs_dir, mp = _tiny_train_dataset(tmp_path)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    cfg = _train_cfg(tmp_path, proc, pairs_dir, mp, epochs=1)
    with pytest.raises(FileNotFoundError):
        train_model(cfg, tmp_path / "empty", torch.device("cpu"), resume=True)


def test_time_budget_stops_before_killing_session(tmp_path, tiny_model):
    """--time-budget-s: dừng graceful để kịp evaluate, không bị Kaggle kill."""
    proc, pairs_dir, mp = _tiny_train_dataset(tmp_path)
    (tmp_path / "cfg.yaml").write_text("x: 1", encoding="utf-8")
    out = tmp_path / "out"
    cfg = _train_cfg(tmp_path, proc, pairs_dir, mp, epochs=5)
    res = train_model(cfg, out, torch.device("cpu"), time_budget_s=0.001)
    assert res["stopped_reason"] == "time_budget"
    assert res["epochs_completed"] < 5
    assert (out / "best_checkpoint.pt").exists()  # vẫn còn checkpoint để evaluate


def test_eval_writes_completion_marker(tmp_path):
    """Notebook dựa vào EVAL_COMPLETE.json để biết chắc eval xong trước khi tar."""
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    m = CodeBERTPairClassifier(HF_ID)
    torch.save({"model_state": m.state_dict()}, out / "best_checkpoint.pt")
    cfg = {
        "dataset": {"manifest": str(mp), "pairs_dir": str(pairs_dir), "processed": str(proc)},
        "model": {"hf_id": HF_ID},
        "tokenizer": {"max_length": 32},
        "inference": {"pair_batch_size": 2},
        "threshold": {"grid_step": 0.1},
        "error_analysis": {"top_k": 10},
    }
    evaluate_model(cfg, out, torch.device("cpu"))
    marker = json.loads((out / "EVAL_COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["completed"] is True and "test" in marker["summary"]


def test_atomic_write_leaves_no_partial_file(tmp_path):
    from src.experiments.exp002_codebert import _atomic_write_text

    p = tmp_path / "a.json"
    _atomic_write_text(p, json.dumps({"v": 1}))
    _atomic_write_text(p, json.dumps({"v": 2}))
    assert json.loads(p.read_text(encoding="utf-8")) == {"v": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_export_eval_weights_is_small_and_weights_only_safe(tmp_path):
    """Bản nhẹ phải nhỏ hơn nhiều, load được bằng `weights_only=True`, và
    `evaluate_model` chạy được khi máy chỉ còn bản nhẹ (không tải 1,4GB về)."""
    proc, pairs_dir, mp = _tiny_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    m = CodeBERTPairClassifier(HF_ID)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
    # step một lần để optimizer state (2 tensor/param) có thật — giống checkpoint
    # sau 31k update, nếu không thì optimizer rỗng và bản nhẹ chẳng nhỏ hơn.
    sum(p.sum() for p in m.parameters()).backward()
    opt.step()
    torch.save({"model_state": m.state_dict(), "optimizer_state": opt.state_dict(),
                "epoch": 1, "best_f1": 0.5,
                "rng": {"numpy": __import__("numpy").random.get_state()}},
               out / "best_checkpoint.pt")

    info = export_eval_weights(out)
    light = out / EVAL_WEIGHTS_NAME
    assert light.exists() and info["to_mb"] < 0.8 * info["from_mb"]

    # weights_only=True mở được => đã loại object lạ (np RNG state) của bản đầy đủ
    loaded = torch.load(light, map_location="cpu", weights_only=True)
    assert set(loaded) == {"model_state", "epoch", "best_f1"}
    for a, b in zip(m.state_dict().values(), loaded["model_state"].values()):
        assert torch.equal(a, b)
    # loader dùng weights_only=True cho bản nhẹ (an toàn) và vẫn đọc được bản đầy đủ
    for k, v in m.state_dict().items():
        assert torch.equal(load_eval_weights(light, "cpu")[k], v)

    # chỉ còn bản nhẹ: resolve vẫn trúng và evaluate vẫn chạy
    (out / "best_checkpoint.pt").unlink()
    assert resolve_eval_checkpoint(out) == light
    cfg = {
        "dataset": {"manifest": str(mp), "pairs_dir": str(pairs_dir), "processed": str(proc)},
        "model": {"hf_id": HF_ID},
        "tokenizer": {"max_length": 32},
        "inference": {"pair_batch_size": 2},
        "threshold": {"grid_step": 0.1},
        "error_analysis": {"top_k": 10},
    }
    summary = evaluate_model(cfg, out, torch.device("cpu"))
    assert "f1" in summary["test"]

    # bản đầy đủ (có np RNG state) vẫn phải nạp được qua cùng loader
    full = out / "best_checkpoint.pt"
    torch.save({"model_state": m.state_dict(),
                "rng": {"numpy": __import__("numpy").random.get_state()}}, full)
    assert resolve_eval_checkpoint(out) == full
    for k, v in m.state_dict().items():
        assert torch.equal(load_eval_weights(full, "cpu")[k], v)


def test_resolve_eval_checkpoint_fails_fast_without_any(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        resolve_eval_checkpoint(tmp_path)
