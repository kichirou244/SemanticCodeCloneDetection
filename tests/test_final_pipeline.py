"""Pipeline cuối (EXP-FINAL-001): xếp hạng, chọn theo val F1, McNemar.

Không cần GPU/Joern — mọi test chỉ đọc artifact JSON đã commit trong `results/`.
Test nào cần artifact mà máy thiếu thì skip, không fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.final_pipeline import _resolve_model
from src.common.config import load_config
from src.evaluation.leaderboard import (
    compare_pair,
    load_runs,
    rank,
    run_selection,
    select,
    significance,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "final_pipeline.yaml"
DEMO_CONFIG = ROOT / "configs" / "demo_web.yaml"

#: Run dùng để kiểm tra bảng xếp hạng — phải có artifact thì mới test được.
NEEDED = ["results/exp003_graphcodebert_poj104_512", "results/exp006_hybrid_poj104_512"]


def _skip_if_no_artifacts() -> None:
    for rel in NEEDED:
        if not (ROOT / rel / "metrics.json").exists():
            pytest.skip(f"thiếu artifact {rel}")


def test_config_has_required_blocks():
    cfg = load_config(CONFIG)
    assert cfg["pipeline"]["id"] == "EXP-FINAL-001"
    assert cfg["selection"]["metric"] == "validation.f1"
    assert cfg["runs"] and cfg["models"]
    ids = [r["id"] for r in cfg["runs"]]
    assert len(ids) == len(set(ids)), "id run phải duy nhất"


def test_every_run_model_key_exists_in_models_block():
    """`model:` của run phải trỏ tới một khoá thật trong `models:` (nếu có)."""
    cfg = load_config(CONFIG)
    for run in cfg["runs"]:
        if run.get("model"):
            assert run["model"] in cfg["models"], run["id"]


def test_inference_default_is_winner_or_known_model():
    cfg = load_config(CONFIG)
    default = cfg["inference"]["default_model"]
    assert default == "winner" or default in cfg["models"]


def test_selection_ranks_by_val_f1_and_picks_best_candidate():
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    rows = load_runs(cfg, ROOT)
    winner, runner, _ = select(rows, cfg)
    assert winner is not None and runner is not None
    assert winner.role == "candidate"
    assert runner.role == "candidate"
    # Winner là candidate có val F1 cao nhất.
    best_val = max(r.value("validation.f1") for r in rows if r.role == "candidate")
    assert winner.value("validation.f1") == pytest.approx(best_val)


def test_ranking_is_monotonic_in_selection_metric():
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    rows = load_runs(cfg, ROOT)
    ordered = rank(rows, "validation.f1", cfg["selection"]["tie_break"])
    vals = [r.value("validation.f1") for r in ordered]
    assert vals == sorted(vals, reverse=True)


def test_baseline_never_selected():
    """MOSS (F1 thấp) không được chọn dù có trong bảng."""
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    winner, _, _ = select(load_runs(cfg, ROOT), cfg)
    assert winner is not None and winner.role == "candidate"
    assert winner.id not in {"B1", "B0"}


def test_thin_threshold_is_warned():
    """M5-512 có ngưỡng 0.01 — pipeline phải nói ra, không im lặng."""
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    winner, _, warnings = select(load_runs(cfg, ROOT), cfg)
    assert winner.threshold is not None
    if winner.threshold < cfg["selection"]["min_threshold"]:
        assert any("mỏng" in w for w in warnings)


def test_significance_combines_both_tests():
    """McNemar và bootstrap có thể bất đồng ⇒ phải trả 'mixed', không chọn bừa."""
    def comp(mc_p, boot_sig):
        return {"mcnemar": {"p_value": mc_p},
                "bootstrap": {"significant": boot_sig}}

    assert significance(comp(0.001, True)) == "yes"
    assert significance(comp(0.4, False)) == "no"
    assert significance(comp(0.4, True)) == "mixed"
    assert significance(comp(0.001, False)) == "mixed"
    assert significance(None) == "unknown"


def test_run_selection_writes_artifacts_and_is_deterministic(tmp_path):
    _skip_if_no_artifacts()
    first = run_selection(CONFIG, root=ROOT, out_dir=tmp_path / "a")
    second = run_selection(CONFIG, root=ROOT, out_dir=tmp_path / "b")
    assert (tmp_path / "a" / "leaderboard.md").exists()
    assert (tmp_path / "a" / "leaderboard.json").exists()
    assert first["winner"]["id"] == second["winner"]["id"]
    assert (tmp_path / "a" / "leaderboard.md").read_text(
        encoding="utf-8"
    ) == (tmp_path / "b" / "leaderboard.md").read_text(encoding="utf-8")


def test_compare_pair_uses_validation_split_and_uncertainty_module():
    """So sánh cặp phải ghép đúng `pair_id` và tái dùng uncertainty.mcnemar."""
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    winner, runner, _ = select(load_runs(cfg, ROOT), cfg)
    comp = compare_pair(winner, runner, ROOT, split="validation", n_boot=0)
    assert comp is not None
    assert comp["split"] == "validation"
    assert comp["a"] == winner.id and comp["b"] == runner.id
    mc = comp["mcnemar"]
    assert comp["n"] == 10000  # validation POJ-104 = 10.000 cặp
    assert 0 < mc["discordant"] <= comp["n"]
    assert 0.0 <= mc["p_value"] <= 1.0
    assert "threshold_a" in mc and "threshold_b" in mc
    assert comp["bootstrap"] is None  # n_boot=0 ⇒ bỏ bootstrap cho nhanh


def test_compare_pair_skips_without_score_files(tmp_path):
    cfg = load_config(CONFIG)
    rows = load_runs(cfg, ROOT)
    if not rows:
        pytest.skip("không có run nào")
    # Trỏ sai thư mục ⇒ không có predictions ⇒ trả None thay vì nổ.
    import copy

    a, b = copy.deepcopy(rows[0]), copy.deepcopy(rows[0])
    a.results_dir = b.results_dir = "results/khong-ton-tai"
    assert compare_pair(a, b, ROOT) is None


def test_resolve_model_defaults_to_config_and_respects_override(tmp_path):
    _skip_if_no_artifacts()
    cfg = load_config(CONFIG)
    selection = run_selection(CONFIG, root=ROOT, out_dir=tmp_path / "sel")
    # default_model đặt thẳng (M2_512) — không phải "winner" thì lấy đúng nó.
    model_key = _resolve_model(cfg, selection, None)
    assert model_key == cfg["inference"]["default_model"]
    assert model_key in cfg["models"]
    # Ép model khác thì tôn trọng override.
    assert _resolve_model(cfg, selection, "M5_512") == "M5_512"


def test_detect_cli_wiring_with_stubbed_registry(tmp_path, monkeypatch, capsys):
    """`detect` phải ghép đúng config → registry → verdict, không cần model thật."""
    import json

    import src.demo.registry as regmod
    from src.demo.scorers.base import ScoreResult

    class Stub:
        label = "stub-model"

        def score(self, code_a, code_b):
            assert code_a == "int main(){}" and code_b == "int main(){}"
            return ScoreResult(ok=True, score=0.9, threshold=0.01,
                               verdict="clone", latency_ms=1.0)

    class FakeRegistry:
        def __init__(self, cfg, root):
            self.cfg = cfg

        def get(self, model_key):
            assert model_key == "M5_512"
            return Stub()

    monkeypatch.setattr(regmod, "ModelRegistry", FakeRegistry)

    a = tmp_path / "a.cpp"
    b = tmp_path / "b.cpp"
    a.write_text("int main(){}", encoding="utf-8")
    b.write_text("int main(){}", encoding="utf-8")

    from scripts.final_pipeline import main

    rc = main(["--out-dir", str(tmp_path / "sel"),
               "detect", "--a", str(a), "--b", str(b), "--model", "M5_512", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "M5_512"
    assert out["verdict"] == "clone" and out["score"] == 0.9


def test_demo_config_untouched_by_final_pipeline():
    """Pipeline cuối không được âm thầm đổi demo (6 model như cũ)."""
    cfg = load_config(DEMO_CONFIG)
    assert cfg["demo"]["order"] == ["B1", "M1", "M2", "M3", "M4", "M5"]
