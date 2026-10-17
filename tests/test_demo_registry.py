"""Registry demo: đủ 6 model, MOSS đúng EXP-013, M4 hỏng êm khi thiếu Java."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.demo.registry import ModelRegistry
from src.demo.scorers.base import ScorerState
from src.demo.scorers.cpg_scorer import CpgScorer
from src.demo.scorers.hybrid_scorer import HybridScorer
from src.demo.scorers.moss_scorer import MossScorer

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs" / "demo_web.yaml"

CODE_A = "int add(int a, int b) { return a + b; }"
CODE_B = "int add(int x, int y) { int s = x + y; return s; }"


def test_registry_builds_all_six_from_yaml():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    assert reg.order == ["B1", "M1", "M2", "M3", "M4", "M5"]
    by_id = {s.model_id: s for s in reg.all()}
    assert isinstance(by_id["B1"], MossScorer)
    assert isinstance(by_id["M4"], CpgScorer)
    assert isinstance(by_id["M5"], HybridScorer)


def test_hybrid_uses_exp006_512_threshold_and_checkpoint():
    """Regression: M5 trỏ về EXP-006 bản 512 (DEC-031) — ngưỡng đọc từ
    threshold.json của run 512, checkpoint là best_checkpoint.pt đầy đủ."""
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    m5 = reg.get("M5")
    assert m5.results_dir.name == "exp006_hybrid_poj104_512"
    assert m5.checkpoint.name == "best_checkpoint.pt"
    assert m5.hf_id == "microsoft/graphcodebert-base"
    assert m5.max_length == 512


def test_transformer_models_use_512_runs():
    """Regression DEC-031: M1/M2 demo phải là bản 512, không được lùi về 256.

    Bản 256 đã đóng băng trong `results/` để đối chiếu, nhưng demo/báo cáo dùng
    bản 512 — nếu ai đó sửa `demo_web.yaml` về 256 thì điểm demo sẽ lệch khỏi
    số đã báo cáo.
    """
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    for mid, run_name, hf_id in (
        ("M1", "exp002_codebert_poj104_512", "microsoft/codebert-base"),
        ("M2", "exp003_graphcodebert_poj104_512", "microsoft/graphcodebert-base"),
    ):
        scorer = reg.get(mid)
        assert scorer.results_dir.name == run_name, mid
        assert scorer.max_length == 512, mid
        assert scorer.hf_id == hf_id, mid
        assert scorer.checkpoint == scorer.results_dir / "best_checkpoint.pt", mid


def test_moss_uses_exp014_service_keeps_fallback_params():
    """Regression: B1 trỏ về baseline MOSS service (EXP-014), còn k/w (20, 5) giữ
    làm tham số của fallback winnowing local khi chưa có MOSS_USER_ID."""
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    moss = reg.get("B1")
    assert moss.results_dir.name == "exp014_moss_service_poj104"
    assert (moss.k, moss.w) == (20, 5)
    assert moss.max_matches == 1000 and moss.cache_dir is not None


def test_moss_scores_end_to_end_with_real_threshold():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    res = reg.get("B1").score(CODE_A, CODE_B)
    assert res.ok and res.threshold is not None
    assert res.verdict in ("clone", "not_clone")  # có ngưỡng thật, không unknown


def test_cpg_scorer_fails_gracefully_without_toolchain(tmp_path, monkeypatch):
    """Giả lập máy thiếu Joern (JOERN_BIN trỏ chỗ không tồn tại): M4 phải báo
    hỏng êm với lý do rõ ràng, không kéo sập request. Deterministic mọi máy
    (không phụ thuộc Java có cài hay không)."""
    monkeypatch.setenv("JOERN_BIN", str(tmp_path / "joern-khong-co"))
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    m4 = reg.get("M4")
    assert m4.ensure_loaded() is False
    assert m4.state == ScorerState.FAILED
    assert m4.status()["error"] and "Joern" in m4.status()["error"]
    res = m4.score(CODE_A, CODE_B)
    assert res.ok is False and res.error


def test_cpg_scorer_loads_with_real_toolchain():
    """Máy có đủ Java + Joern + checkpoint + torch_geometric thì M4 nạp được.
    Thiếu thứ gì thì skip (máy CI không có Java)."""
    pytest.importorskip("torch_geometric")
    from scripts.build_cpg_export import joern_missing_reason, joern_or_die

    if joern_missing_reason(joern_or_die()):
        pytest.skip("cần Java+Joern để nạp M4 thật")
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    m4 = reg.get("M4")
    assert m4.ensure_loaded() is True, m4.status()["error"]
    assert m4.state == ScorerState.READY


def test_cpg_scorer_scores_real_pair_with_joern():
    """Bằng chứng cuối: Joern thật dựng CPG cho 2 hàm rồi GAT chấm ra số.
    Tốn ~1-3 phút (JVM). Skip khi thiếu toolchain."""
    pytest.importorskip("torch_geometric")
    from scripts.build_cpg_export import joern_missing_reason, joern_or_die

    if joern_missing_reason(joern_or_die()):
        pytest.skip("cần Java+Joern để chấm M4 thật")
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    res = reg.get("M4").score(CODE_A, CODE_B)
    assert res.ok, res.error
    assert res.threshold is not None
    assert 0.0 <= res.score <= 1.0
    assert res.verdict in ("clone", "not_clone")
    assert res.detail.get("n_nodes_a", 0) > 10


def test_hybrid_scorer_fails_gracefully_without_toolchain(tmp_path, monkeypatch):
    """M5 như M4: thiếu Joern thì hỏng êm, không kéo sập request."""
    monkeypatch.setenv("JOERN_BIN", str(tmp_path / "joern-khong-co"))
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    m5 = reg.get("M5")
    assert m5.ensure_loaded() is False
    assert m5.state == ScorerState.FAILED
    assert m5.status()["error"] and "Joern" in m5.status()["error"]
    res = m5.score(CODE_A, CODE_B)
    assert res.ok is False and res.error


def test_hybrid_scorer_scores_real_pair_with_joern():
    """Joern thật dựng CPG + GraphCodeBERT tokenize rồi hybrid chấm ra số.
    Tốn vài phút (JVM + model 128M). Skip khi thiếu toolchain/checkpoint."""
    pytest.importorskip("torch_geometric")
    from scripts.build_cpg_export import joern_missing_reason, joern_or_die

    if joern_missing_reason(joern_or_die()):
        pytest.skip("cần Java+Joern để chấm M5 thật")
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    if not reg.get("M5").checkpoint.exists():
        pytest.skip("thiếu checkpoint M5")
    res = reg.get("M5").score(CODE_A, CODE_B)
    assert res.ok, res.error
    assert res.threshold is not None
    assert 0.0 <= res.score <= 1.0
    assert res.verdict in ("clone", "not_clone")
    assert res.detail.get("n_nodes_a", 0) > 10


def test_unknown_model_id_raises_keyerror():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    try:
        reg.get("M9")
    except KeyError:
        return
    raise AssertionError("phải KeyError với id lạ")
