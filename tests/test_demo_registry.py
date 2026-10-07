"""Registry demo: đủ 5 model, MOSS đúng EXP-013, M4 hỏng êm khi thiếu Java."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.demo.registry import ModelRegistry
from src.demo.scorers.base import ScorerState
from src.demo.scorers.cpg_scorer import CpgScorer
from src.demo.scorers.moss_scorer import MossScorer

ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "configs" / "demo_web.yaml"

CODE_A = "int add(int a, int b) { return a + b; }"
CODE_B = "int add(int x, int y) { int s = x + y; return s; }"


def test_registry_builds_all_five_from_yaml():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    assert reg.order == ["B1", "M1", "M2", "M3", "M4"]
    by_id = {s.model_id: s for s in reg.all()}
    assert isinstance(by_id["B1"], MossScorer)
    assert isinstance(by_id["M4"], CpgScorer)


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


def test_unknown_model_id_raises_keyerror():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    try:
        reg.get("M9")
    except KeyError:
        return
    raise AssertionError("phải KeyError với id lạ")
