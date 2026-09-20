"""Tests EXP-002 (CPU): model, tokenizer, checkpoint, threshold, leakage, runner (§36).

Chạy trên máy user (cần checkpoint microsoft/codebert-base trong HF cache —
đã có từ smoke_transformer ở INIT).
"""

import json

import torch

from src.experiments.exp002_codebert import (
    evaluate_model,
    load_code_map,
    load_pair_refs,
    preflight,
    truncation_analysis,
)
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
