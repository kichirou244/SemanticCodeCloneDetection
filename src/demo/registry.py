"""Dựng `BaseScorer` từ `configs/demo_web.yaml`.

Registry là nơi duy nhất biết model nào tồn tại và cần gì để chạy. Thêm M4/M5 sau
này = thêm một khối trong YAML, không sửa service hay server.

Đường dẫn trong YAML được hiểu là **tương đối với thư mục gốc repo**. Cần vậy vì
cùng một file config được dùng khi chạy từ gốc, từ `scripts/`, và trong test.
"""

from __future__ import annotations

from pathlib import Path

from src.common.config import load_config
from src.demo.scorers.astnn_scorer import AstnnScorer
from src.demo.scorers.base import BaseScorer
from src.demo.scorers.cpg_scorer import CpgScorer
from src.demo.scorers.moss_scorer import MossScorer
from src.demo.scorers.transformer_scorer import TransformerScorer
from src.models.baseline.moss import DEFAULT_K, DEFAULT_W

DEFAULT_CONFIG = Path("configs/demo_web.yaml")

#: `kind` trong YAML → class. Thêm loại scorer mới thì thêm ở đây, không chạm chỗ khác.
_BUILDERS = {
    "moss": MossScorer,
    "transformer": TransformerScorer,
    "astnn": AstnnScorer,
    "cpg": CpgScorer,
}


class ModelRegistry:
    """Giữ danh sách scorer theo đúng thứ tự hiển thị, dựng lười (không nạp)."""

    def __init__(self, config: dict, root: Path):
        self.config = config
        self.root = root
        self._scorers: dict[str, BaseScorer] = {}

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_CONFIG, root: Path | None = None) -> "ModelRegistry":
        cfg_path = Path(path)
        base = Path(root) if root else Path.cwd()
        config = load_config(cfg_path)
        return cls(config, base)

    # ------------------------------------------------------------------ truy vấn
    @property
    def order(self) -> list[str]:
        return list(self.config.get("demo", {}).get("order", sorted(self.config["models"])))

    def all(self) -> list[BaseScorer]:
        return [self.get(mid) for mid in self.order]

    def get(self, model_id: str) -> BaseScorer:
        """Lấy scorer, dựng nếu chưa. Id lạ → `KeyError` (config sai, phải fail)."""
        if model_id not in self._scorers:
            spec = self.config["models"].get(model_id)
            if spec is None:
                raise KeyError(f"không có model {model_id!r} trong registry")
            self._scorers[model_id] = _build(spec, model_id, self.root,
                                             self.config.get("runtime", {}).get("device", "auto"))
        return self._scorers[model_id]

    def selected(self, model_ids: list[str] | None) -> list[BaseScorer]:
        """Danh sách scorer theo yêu cầu; `None`/rỗng ⇒ toàn bộ theo thứ tự config."""
        if not model_ids:
            return self.all()
        return [self.get(mid) for mid in model_ids if mid in self.config["models"]]

    def preload(self, model_ids: list[str]) -> dict[str, str]:
        """Nạp trước một danh sách model, trả `{model_id: "ready" | lỗi}`.

        Nạp tuần tự chứ không song song: torch + 3 model 126M tham số cùng lúc sẽ
        tranh GPU/CPU và dễ OOM. Mỗi cái vài chục giây, chấp nhận được.
        """
        out: dict[str, str] = {}
        for mid in model_ids or []:
            scorer = self.get(mid)
            out[mid] = "ready" if scorer.ensure_loaded() else (scorer.status()["error"] or "failed")
        return out


def _build(spec: dict, model_id: str, root: Path, device: str) -> BaseScorer:
    kind = spec["kind"]
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"{model_id}: kind {kind!r} chưa hỗ trợ "
                         f"(có: {', '.join(sorted(_BUILDERS))})")
    common = {
        "model_id": model_id,
        "label": spec.get("label", model_id),
        "results_dir": _p(root, spec["results_dir"]),
        "device": device,
    }
    if kind == "moss":
        return MossScorer(
            **common,
            k=int(spec.get("k", DEFAULT_K)),
            w=int(spec.get("w", DEFAULT_W)),
        )
    if kind == "transformer":
        return TransformerScorer(
            **common,
            checkpoint=_p(root, spec["checkpoint"]),
            hf_id=spec["hf_id"],
            max_length=int(spec.get("max_length", 256)),
        )
    if kind == "cpg":
        return CpgScorer(
            **common,
            checkpoint=_p(root, spec["checkpoint"]),
            hidden_size=int(spec.get("hidden_size", 128)),
            num_layers=int(spec.get("num_layers", 3)),
            heads=int(spec.get("heads", 4)),
            dropout=float(spec.get("dropout", 0.2)),
            max_nodes=int(spec.get("max_nodes", 1500)),
        )
    return AstnnScorer(
        **common,
        checkpoint=_p(root, spec["checkpoint"]),
        hidden_size=int(spec.get("hidden_size", 200)),
        dropout=float(spec.get("dropout", 0.2)),
        language=spec.get("language", "cpp"),
        max_code_chars=int(spec.get("max_code_chars", 20000)),
    )


def _p(root: Path, rel: str) -> Path:
    """Chuẩn hoá đường dẫn tương đối về tuyệt đối theo thư mục gốc repo."""
    path = Path(rel)
    return path if path.is_absolute() else root / path
