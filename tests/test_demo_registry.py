"""Registry demo: đủ 5 model, MOSS đúng EXP-013, M4 hỏng êm khi thiếu Java."""

from __future__ import annotations

from pathlib import Path

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


def test_moss_uses_exp013_tuned_params_not_defaults():
    """Regression: YAML từng trỏ EXP-006 (không tồn tại) với k=50/w=100 —
    bộ suy biến trên POJ-104. Phải là EXP-013 (k=20, w=5)."""
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    moss = reg.get("B1")
    assert moss.results_dir.name == "exp013_moss_poj104"
    assert (moss.k, moss.w) == (20, 5)


def test_moss_scores_end_to_end_with_real_threshold():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    res = reg.get("B1").score(CODE_A, CODE_B)
    assert res.ok and res.threshold is not None
    assert res.verdict in ("clone", "not_clone")  # có ngưỡng thật, không unknown


def test_cpg_scorer_fails_gracefully_without_java():
    """Máy này không có Java (DEC-023) — đúng môi trường để kiểm tra M4 báo
    hỏng êm thay vì kéo sập request."""
    import shutil

    assert shutil.which("java") is None, "máy này có Java thì test mất ý nghĩa"
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    m4 = reg.get("M4")
    assert m4.ensure_loaded() is False
    assert m4.state == ScorerState.FAILED
    assert m4.status()["error"] and "Joern" in m4.status()["error"]
    res = m4.score(CODE_A, CODE_B)
    assert res.ok is False and res.error


def test_unknown_model_id_raises_keyerror():
    reg = ModelRegistry.from_yaml(YAML, root=ROOT)
    try:
        reg.get("M9")
    except KeyError:
        return
    raise AssertionError("phải KeyError với id lạ")
