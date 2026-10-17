"""Bảng xếp hạng + chọn phương pháp cho pipeline cuối (EXP-FINAL-001).

Khác `metrics.py`: module này **không tính lại metric từ score**. Nó đọc artifact
đã đóng băng của từng experiment (`metrics.json`, `threshold.json`,
`predictions/*_scores.jsonl`) rồi xếp hạng, chọn và so sánh. Nhờ vậy bước
"đánh giá & chọn phương pháp" chạy được mà **không cần torch/GPU/Joern**.

Nguyên tắc (theo `docs/evaluation_protocol.md` và DEC-022):

- Ngưỡng quyết định của mỗi run là ngưỡng **đã chọn trên validation** và đóng
  băng trong `threshold.json`; pipeline không bao giờ chọn lại ngưỡng trên test.
- Chọn phương pháp dựa trên **val F1**, không nhìn test — chọn theo test là tự
  khen (test bị dùng để báo cáo, không dùng để quyết định).
- Cảnh báo chỉ *nêu ra*, không tự loại ứng viên: quyết định cuối là của con người.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import load_config
from src.evaluation.uncertainty import bootstrap_paired, mcnemar

DEFAULT_CONFIG = Path("configs/final_pipeline.yaml")
DEFAULT_OUT = Path("results/final_pipeline")

#: Ngưỡng heuristic coi là "mỏng" — dưới mức này verdict rất nhạy với nhiễu
#: hiệu chỉnh xác suất (M5-512 có ngưỡng 0.01). Chỉ để cảnh báo, không loại.
THIN_THRESHOLD = 0.05


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class RunResult:
    """Một experiment đã hoàn thành, đủ artifact để so sánh."""

    id: str
    label: str
    family: str
    role: str  # "baseline" | "candidate"
    results_dir: str
    #: Khoá trong khối `models:` của config, dùng cho bước `detect`. Rỗng nghĩa
    #: là run này chỉ để so sánh số, chưa gắn model chạy được.
    model: str = ""
    metrics: dict = field(default_factory=dict)
    threshold: float | None = None
    notes: str = ""

    def value(self, dotted: str) -> float | None:
        """Đọc metric theo đường dẫn `split.metric`, ví dụ `validation.f1`."""
        node: Any = self.metrics
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return float(node) if isinstance(node, (int, float)) else None

    def to_dict(self) -> dict:
        return asdict(self)


def load_run(spec: dict, root: Path) -> RunResult:
    """Nạp một run từ khai báo trong config + artifact trên đĩa."""
    rdir = root / spec["results_dir"]
    metrics = read_json(rdir / "metrics.json") or {}
    thr_file = read_json(rdir / "threshold.json") or {}
    threshold = metrics.get("validation", {}).get("threshold")
    if threshold is None:
        threshold = thr_file.get("threshold")
    return RunResult(
        id=spec["id"],
        label=spec.get("label", spec["id"]),
        family=spec.get("family", ""),
        role=spec.get("role", "candidate"),
        results_dir=spec["results_dir"],
        model=spec.get("model", ""),
        metrics=metrics,
        threshold=float(threshold) if threshold is not None else None,
        notes=spec.get("notes", ""),
    )


def load_runs(cfg: dict, root: Path) -> list[RunResult]:
    return [load_run(spec, root) for spec in cfg.get("runs", [])]


def _sort_key(row: RunResult, metric: str, tie_break: list[str]):
    vals = [row.value(metric)] + [row.value(t) for t in tie_break]
    return tuple(v if v is not None else float("-inf") for v in vals)


def rank(
    rows: list[RunResult], metric: str, tie_break: list[str] | None = None
) -> list[RunResult]:
    """Xếp hạng giảm dần theo `metric`, phá hoà bằng `tie_break`."""
    tb = tie_break or []
    return sorted(rows, key=lambda r: _sort_key(r, metric, tb), reverse=True)


# --------------------------------------------------------------- so sánh cặp
def load_scores(path: Path) -> dict[str, tuple[int, float]]:
    """Đọc `predictions/*_scores.jsonl` → `{pair_id: (label, score)}`."""
    out: dict[str, tuple[int, float]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row["pair_id"])] = (int(row["label"]), float(row["score"]))
    return out


def compare_pair(
    winner: RunResult,
    runner: RunResult,
    root: Path,
    split: str = "validation",
    n_boot: int = 500,
    seed: int = 42,
) -> dict | None:
    """So sánh cặp winner vs á quân trên `split` — tái dùng `uncertainty.py`.

    KHÔNG tự viết lại McNemar/bootstrap: `src/evaluation/uncertainty.py` đã có
    bản exact-binomial (chống tràn) và paired bootstrap (mỗi model một ngưỡng
    ghim riêng). Ở đây chỉ ghép hai bộ score theo `pair_id` chung rồi gọi lại.
    """
    if winner.threshold is None or runner.threshold is None:
        return None
    a = load_scores(root / winner.results_dir / "predictions" / f"{split}_scores.jsonl")
    b = load_scores(root / runner.results_dir / "predictions" / f"{split}_scores.jsonl")
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    y = np.asarray([a[p][0] for p in shared], dtype=np.int8)
    sa = np.asarray([a[p][1] for p in shared], dtype=float)
    sb = np.asarray([b[p][1] for p in shared], dtype=float)
    mc = mcnemar(y, sa, sb, winner.threshold, runner.threshold)
    boot = (
        bootstrap_paired(y, sa, sb, winner.threshold, runner.threshold,
                         n_boot=n_boot, seed=seed)
        if n_boot > 0 else None
    )
    return {
        "split": split,
        "a": winner.id,
        "b": runner.id,
        "n": len(shared),
        "mcnemar": mc,
        "bootstrap": boot,
        "significant_at_0.05": bool(mc["p_value"] < 0.05),
    }


def significance(comparison: dict | None) -> str:
    """Kết luận tổng hợp McNemar + bootstrap ghép: `yes` / `no` / `mixed`.

    Hai kiểm định đo hai thứ khác nhau (McNemar: tỉ lệ cặp bất đồng ý; bootstrap:
    độ lớn ΔF1), nên chúng **có thể bất đồng** — khi đó phải nói `mixed` chứ
    không được chọn bừa một cái để kết luận cho gọn.
    """
    if not comparison:
        return "unknown"
    mc_sig = comparison["mcnemar"]["p_value"] < 0.05
    boot = comparison.get("bootstrap")
    boot_sig = bool(boot["significant"]) if boot else mc_sig
    if mc_sig and boot_sig:
        return "yes"
    if not mc_sig and not boot_sig:
        return "no"
    return "mixed"


# ------------------------------------------------------------------- chọn
def select(
    rows: list[RunResult], cfg: dict
) -> tuple[RunResult | None, RunResult | None, list[str]]:
    """Chọn winner theo luật trong config; trả `(winner, runner_up, warnings)`."""
    rule = cfg.get("selection", {})
    metric = rule.get("metric", "validation.f1")
    tie_break = rule.get("tie_break", []) or []
    min_lead = float(rule.get("min_lead_over_runner_up", 0.01))
    min_thr = float(rule.get("min_threshold", THIN_THRESHOLD))

    candidates = [r for r in rows if r.role == "candidate"]
    ranked = rank(candidates, metric, tie_break)
    if not ranked:
        return None, None, ["Không có run nào role=candidate để chọn."]

    winner = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    warnings: list[str] = []

    if winner.threshold is not None and winner.threshold < min_thr:
        warnings.append(
            f"Ngưỡng của {winner.id} rất mỏng ({winner.threshold:g} < {min_thr:g}) "
            "— verdict nhạy với nhiễu hiệu chỉnh xác suất."
        )
    if runner is not None:
        wv, rv = winner.value(metric), runner.value(metric)
        if wv is not None and rv is not None and (wv - rv) < min_lead:
            warnings.append(
                f"{metric} của {winner.id} ({wv:.4f}) chỉ hơn {runner.id} "
                f"({rv:.4f}) đúng {wv - rv:.4f} (< {min_lead:g}) — rất sát."
            )

    # Chọn theo val mà người thắng test lại khác ⇒ nói rõ để không tự huyễn.
    by_test = rank(candidates, "test.f1", ["test.roc_auc"])
    if by_test and by_test[0].id != winner.id:
        warnings.append(
            f"Theo test F1 thì {by_test[0].id} "
            f"({by_test[0].value('test.f1'):.4f}) lại hơn {winner.id} "
            f"({winner.value('test.f1'):.4f}) — nhưng test không dùng để chọn."
        )
    return winner, runner, warnings


# -------------------------------------------------------------- kết xuất
def render_markdown(rows: list[RunResult], winner_id: str | None, cfg: dict) -> str:
    rule = cfg.get("selection", {})
    metric = rule.get("metric", "validation.f1")
    ordered = rank(rows, metric, rule.get("tie_break", []) or [])
    lines = [
        "# Bảng xếp hạng pipeline cuối (EXP-FINAL-001)",
        "",
        f"- Dataset: **{cfg.get('pipeline', {}).get('dataset', '?')}**",
        (
            f"- Luật chọn: `{metric}` (giảm dần) — ngưỡng đóng băng theo "
            "`threshold.json`, chọn trên validation."
        ),
        f"- Ứng viên được chọn: {winner_id or '—'}",
        "",
        "| # | Vai trò | PP | Val F1 | Test P | Test R | Test F1 | Test AUC | Ngưỡng |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(v: float | None) -> str:
        return "—" if v is None else f"{v:.4f}"

    for i, row in enumerate(ordered, 1):
        mark = "**" if row.id == winner_id else ""
        thr = "—" if row.threshold is None else f"{row.threshold:g}"
        lines.append(
            f"| {i} | {'Ứng viên' if row.role == 'candidate' else 'Baseline'} | "
            f"{mark}{row.id}{mark} | {fmt(row.value('validation.f1'))} | "
            f"{fmt(row.value('test.precision'))} | {fmt(row.value('test.recall'))} | "
            f"{fmt(row.value('test.f1'))} | {fmt(row.value('test.roc_auc'))} | {thr} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(
    out_dir: Path,
    config_path: Path,
    rows: list[RunResult],
    winner: RunResult | None,
    runner: RunResult | None,
    warnings: list[str],
    comparison: dict | None,
    cfg: dict,
) -> dict:
    """Ghi `leaderboard.json/.md` + `selection.json` vào `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = rank(rows, cfg.get("selection", {}).get("metric", "validation.f1"),
                   cfg.get("selection", {}).get("tie_break", []) or [])
    leaderboard = {
        "pipeline": cfg.get("pipeline", {}),
        "selection_rule": cfg.get("selection", {}),
        "config": str(config_path),
        "rows": [r.to_dict() for r in ordered],
    }
    (out_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "leaderboard.md").write_text(
        render_markdown(rows, winner.id if winner else None, cfg), encoding="utf-8"
    )

    selection = {
        "pipeline": cfg.get("pipeline", {}),
        "selection_rule": cfg.get("selection", {}),
        "winner": winner.to_dict() if winner else None,
        "runner_up": runner.to_dict() if runner else None,
        "warnings": warnings,
        "comparison": comparison,
    }
    (out_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return selection


def run_selection(
    config_path: Path | str = DEFAULT_CONFIG,
    root: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Đường chạy đầy đủ: nạp config → xếp hạng → chọn → so sánh → ghi file."""
    root = Path(root) if root else Path.cwd()
    config_path = Path(config_path)
    cfg = load_config(config_path)
    rows = load_runs(cfg, root)
    winner, runner, warnings = select(rows, cfg)
    comparison = None
    if winner is not None and runner is not None:
        rule = cfg.get("selection", {})
        comparison = compare_pair(
            winner, runner, root,
            split=rule.get("comparison_split", "validation"),
            n_boot=int(rule.get("comparison_n_boot", 500)),
            seed=int(cfg.get("pipeline", {}).get("seed", 42)),
        )
    if comparison is not None:
        comparison["significance"] = significance(comparison)
        p = comparison["mcnemar"]["p_value"]
        p_str = "<1e-300" if p == 0.0 else f"{p:.3g}"
        if comparison["significance"] == "no":
            warnings.append(
                f"McNemar {winner.id} vs {runner.id} trên {comparison['split']} "
                f"p = {p_str} (≥ 0,05) và CI bootstrap chứa 0 — khác biệt chưa có "
                "ý nghĩa thống kê; muốn kết luận chắc hơn thì chọn phương pháp đơn "
                "giản/ổn định hơn hoặc chạy thêm seed."
            )
        elif comparison["significance"] == "mixed":
            warnings.append(
                f"Hai kiểm định {winner.id} vs {runner.id} KHÔNG nhất quán "
                f"(McNemar p = {p_str}; bootstrap ghép "
                f"{'có' if comparison['bootstrap']['significant'] else 'không'} "
                "khác biệt) — chênh lệch nhỏ và sát ngưỡng, cần thận trọng khi kết luận."
            )
    return write_artifacts(
        Path(out_dir) if out_dir else root / DEFAULT_OUT,
        config_path, rows, winner, runner, warnings, comparison, cfg,
    )
