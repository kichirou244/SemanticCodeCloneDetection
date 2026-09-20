"""Tests EXP-001: fit-isolation, sparse, cosine, threshold, smoke runner (§25)."""

import json

import numpy as np
import pytest
from scipy import sparse

from src.data.access import stream_codes, stream_pairs
from src.experiments.exp001_tfidf import (
    build_vectorizer,
    score_pairs,
    select_threshold,
    transform_split,
    validate_dataset,
    run_experiment,
)


@pytest.fixture
def tiny_processed(tmp_path):
    p = tmp_path / "samples.jsonl"
    rows = [
        {"sample_id": f"tr-{i}", "code": f"int func{i}(int a){{return a+{i};}}", "problem_id": "p1", "split": "train"}
        for i in range(6)
    ]
    rows += [
        {"sample_id": "va-0", "code": "int func0(int a){return a+0;}", "problem_id": "p2", "split": "validation"},
        {"sample_id": "va-1", "code": "void other(){printf(\"zzz_qqx\");}", "problem_id": "p2", "split": "validation"},
        {"sample_id": "te-0", "code": "int func0(int a){return a+0;}", "problem_id": "p3", "split": "test"},
        {"sample_id": "te-1", "code": "void other(){printf(\"zzz_qqx\");}", "problem_id": "p3", "split": "test"},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_stream_codes_split_filter(tiny_processed):
    assert len(list(stream_codes(tiny_processed, "train"))) == 6
    assert len(list(stream_codes(tiny_processed, "test"))) == 2


def test_fit_only_train_vocab(tiny_processed):
    """Token chỉ xuất hiện ở val/test không được vào vocabulary (§6)."""
    vec = build_vectorizer(1000)
    vec.fit(code for _, code in stream_codes(tiny_processed, "train"))
    assert "zzz_qqx" not in vec.vocabulary_  # chỉ có ở val/test
    assert "func0" in vec.vocabulary_


def test_sparse_and_deterministic(tiny_processed):
    vec = build_vectorizer(1000)
    vec.fit(code for _, code in stream_codes(tiny_processed, "train"))
    idx, mat, _ = transform_split(vec, tiny_processed, "train")
    assert sparse.issparse(mat)
    assert mat.shape[0] == 6 and len(idx) == 6
    _, mat2, _ = transform_split(vec, tiny_processed, "train")
    assert (mat != mat2).nnz == 0  # deterministic


def test_cosine_identical_and_orthogonal(tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    mat = vec.fit_transform(["alpha beta", "alpha beta", "zzz qqq"]).tocsr()
    idx = {"a": 0, "b": 1, "c": 2}
    p = tmp_path / "pairs.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, (a, b) in enumerate([("a", "b"), ("a", "c")]):
            f.write(json.dumps({"pair_id": str(i), "code_a_id": a, "code_b_id": b, "label": 1}) + "\n")
    s, y, _, _ = score_pairs(p, idx, mat, 4096)
    assert abs(s[0] - 1.0) < 1e-6  # identical → ≈1
    assert abs(s[1] - 0.0) < 1e-6  # orthogonal → ≈0


def test_threshold_from_validation_only():
    val_scores = np.array([0.9, 0.8, 0.2, 0.1])
    val_labels = np.array([1, 1, 0, 0])
    thr = select_threshold(val_scores, val_labels)
    assert thr["split"] == "validation" and thr["method"] == "validation_f1"
    assert 0.0 < thr["threshold"] < 1.0


def test_validate_aborts_without_manifest(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path / "no.json", tmp_path, tmp_path / "no2.jsonl")


def _write_pairs(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_smoke_runner(tmp_path, tiny_processed):
    """Smoke full pipeline trên fixture tiny (§26)."""
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    _write_pairs(pairs_dir / "train.jsonl", [
        {"pair_id": "t0", "code_a_id": "tr-0", "code_b_id": "tr-1",
         "problem_a": "p1", "problem_b": "p1", "label": 1, "split": "train"},
        {"pair_id": "t1", "code_a_id": "tr-0", "code_b_id": "tr-2",
         "problem_a": "p1", "problem_b": "p1", "label": 1, "split": "train"},
    ])
    _write_pairs(pairs_dir / "validation.jsonl", [
        {"pair_id": "v0", "code_a_id": "va-0", "code_b_id": "va-0",
         "problem_a": "p2", "problem_b": "p2", "label": 1, "split": "validation"},
        {"pair_id": "v1", "code_a_id": "va-0", "code_b_id": "va-1",
         "problem_a": "p2", "problem_b": "p2", "label": 0, "split": "validation"},
    ])
    _write_pairs(pairs_dir / "test.jsonl", [
        {"pair_id": "e0", "code_a_id": "te-0", "code_b_id": "te-0",
         "problem_a": "p3", "problem_b": "p3", "label": 1, "split": "test"},
        {"pair_id": "e1", "code_a_id": "te-0", "code_b_id": "te-1",
         "problem_a": "p3", "problem_b": "p3", "label": 0, "split": "test"},
    ])
    manifest = {
        "dataset": "tiny", "leakage_passed": True,
        "splits": {"train": {}, "validation": {}, "test": {}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("dummy: 1", encoding="utf-8")
    cfg = {
        "training": {"seed": 42},
        "dataset": {"manifest": str(manifest_path), "pairs_dir": str(pairs_dir),
                    "processed": str(tiny_processed)},
        "model": {"max_features": 1000},
        "inference": {"pair_batch_size": 2},
        "threshold": {"grid_step": 0.05},
        "error_analysis": {"top_k": 10},
    }
    out = tmp_path / "out"
    summary = run_experiment(cfg, cfg_path, out)
    for name in ("metrics.json", "metrics.csv", "threshold.json",
                 "confusion_matrix.json", "error_cases.jsonl",
                 "predictions_test_scores.jsonl", "tfidf_stats.json",
                 "run_metadata.json", "config.yaml", "manifest.json"):
        assert (out / name).exists(), name
    assert 0.0 <= summary["threshold"] <= 1.0
    assert set(summary["test"]) >= {"precision", "recall", "f1", "TP", "TN", "FP", "FN"}
