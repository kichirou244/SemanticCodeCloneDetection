"""Ghép kết quả: căn chỉnh + chấm điểm + trạng thái model.

Đây là tầng mà UI nói chuyện. Nó giữ đúng ba quy tắc để demo không thành nơi sinh
ra con số lệch với báo cáo:

- Ngưỡng lấy từ `threshold.json` của từng experiment, không tự chọn (xem `base`).
- Có model lỗi thì trả lỗi **cho model đó**, không làm hỏng cả response.
- Kiểm tra kích thước input ở đây, không ở FastAPI, để test gọi trực tiếp
  `DemoService` cũng được kiểm tra.
"""

from __future__ import annotations

import io
import time
import uuid
import zipfile
from pathlib import Path

from src.demo.alignment.aligner import align
from src.demo.alignment.lexer import lex_code
from src.demo.registry import ModelRegistry
from src.demo.samples import CodeIndex

VALID_MODES = ("literal", "abstract")

#: Loại file được quét trong chế độ "database". Không nhận .txt — tránh nhiễu.
_LIBRARY_EXTS = frozenset({
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".cu", ".ino",
})


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


class InputTooLarge(ValueError):
    """Input vượt giới hạn — client sửa được, nên trả 4xx chứ không phải 500."""


class DemoService:
    def __init__(
        self,
        registry: ModelRegistry,
        max_code_chars: int = 20000,
        max_lines: int = 400,
        max_block_colors: int = 6,
    ):
        self.registry = registry
        self.limits = {
            "max_code_chars": max_code_chars,
            "max_lines": max_lines,
        }
        self.max_block_colors = max_block_colors
        self._code_index: CodeIndex | None = self._maybe_code_index()
        data_cfg = self.registry.config.get("data", {})
        self._library_dir: Path | None = (
            self.registry.root / data_cfg["library_dir"]
            if data_cfg.get("library_dir")
            else None
        )
        self._library_max_files = int(data_cfg.get("library_max_files", 300))
        self._library_max_chars = int(data_cfg.get("library_max_chars", 20000))
        self._library_default_threshold = float(data_cfg.get("library_default_threshold", 0.5))
        scan_cfg = self.registry.config.get("scan", {})
        self._scan_max_files = int(scan_cfg.get("max_files", 20))
        self._scan_max_bytes = int(scan_cfg.get("max_bytes", 10485760))
        self._scan_max_file_chars = int(scan_cfg.get("max_file_chars", 20000))
        self._upload_root = self.registry.root / "artifacts" / "scan_uploads"

    def _maybe_code_index(self) -> CodeIndex | None:
        """Chỉ tạo `CodeIndex` khi `samples.jsonl` thật sự tồn tại.

        Test và lần chạy đầu tiên (chưa có `data/`) không nên vỡ vì thiếu dữ liệu
        thật — cặp mẫu viết tay vẫn dùng được.
        """
        samples = self._data_path("processed")
        return CodeIndex(samples) if samples and samples.exists() else None

    def _data_path(self, key: str) -> Path | None:
        rel = self.registry.config.get("data", {}).get(key)
        return self.registry.root / rel if rel else None

    # --------------------------------------------------------------- trạng thái
    def status(self) -> dict:
        return {
            "limits": {**self.limits, "n_block_colors": self.max_block_colors},
            "models": [s.status() for s in self.registry.all()],
        }

    # ---------------------------------------------------------------- phân tích
    def analyze(
        self,
        code_a: str,
        code_b: str,
        model_ids: list[str] | None = None,
        mode: str = "abstract",
        min_block_tokens: int = 3,
    ) -> dict:
        """Trả về `{alignment, models, meta}` cho một cặp code."""
        self._validate(code_a, "code_a")
        self._validate(code_b, "code_b")
        if mode not in VALID_MODES:
            raise ValueError(f"mode {mode!r} không hợp lệ (có: {', '.join(VALID_MODES)})")

        alignment_cfg = self.registry.config.get("alignment", {})
        tokens = min_block_tokens or int(alignment_cfg.get("min_block_tokens", 2))
        lines_a, lines_b = lex_code(code_a), lex_code(code_b)
        result = align(lines_a, lines_b, mode=mode, min_block_tokens=tokens)
        return {
            "alignment": result.to_payload(lines_a, lines_b),
            "models": self._score_all(code_a, code_b, model_ids),
            "meta": {
                "mode": mode,
                "n_lines_a": len(lines_a),
                "n_lines_b": len(lines_b),
                "n_block_colors": self.max_block_colors,
            },
        }

    def _score_all(
        self, code_a: str, code_b: str, model_ids: list[str] | None
    ) -> list[dict]:
        out = []
        for scorer in self.registry.selected(model_ids):
            result = scorer.score(code_a, code_b)
            out.append(
                {
                    "model_id": scorer.model_id,
                    "label": scorer.label,
                    "state": scorer.state,
                    "hint": scorer.checkpoint_hint(),
                    **result.to_dict(),
                }
            )
        return out

    # ------------------------------------------------------------------ cặp mẫu
    def real_pairs(self, split: str = "test", limit: int | None = None) -> list[dict]:
        from src.demo.samples import list_real_pairs

        pairs_dir = self._data_path("pairs_dir")
        if pairs_dir is None:
            return []
        n = limit if limit is not None else int(
            self.registry.config.get("data", {}).get("real_pairs_limit", 5)
        )
        return list_real_pairs(pairs_dir, split=split, limit=n)

    def load_real(self, pair_id: str, split: str = "test") -> dict:
        from src.demo.samples import load_real_pair, sample_to_dict

        pairs_dir, index = self._data_path("pairs_dir"), self._code_index
        if pairs_dir is None or index is None:
            raise FileNotFoundError("chưa cấu hình data/pairs/poj104 hoặc samples.jsonl")
        return sample_to_dict(load_real_pair(pair_id, index, pairs_dir, split))

    # ---------------------------------------------------------- quét thư mục
    def library_files(self) -> dict:
        """Liệt kê file code trong `data/library_dir` để UI hiển thị database.

        Trả `{enabled, directory, files: [...]}`; khi chưa khai `library_dir` thì
        `enabled=False` và server vẫn trả JSON hợp lệ (không 404) để UI biết ẩn
        chức năng.
        """
        if self._library_dir is None:
            return {"enabled": False, "directory": None, "files": []}
        root, files = self._library_dir.resolve(), []
        if root.exists():
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in _LIBRARY_EXTS:
                    continue
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                files.append({
                    "path": rel.as_posix(),
                    "name": path.name,
                    "size": path.stat().st_size,
                    "ext": path.suffix.lower(),
                    "n_lines": _count_lines(path),
                })
        files.sort(key=lambda f: f["path"])
        return {
            "enabled": True,
            "directory": str(root),
            "n_files": len(files),
            "max_files": self._library_max_files,
            "max_chars": self._library_max_chars,
            "default_threshold": self._library_default_threshold,
            "files": files,
        }

    def read_library_file(self, rel_path: str) -> dict:
        """Đọc nội dung một file trong library (có kiểm tra an toàn đường dẫn).

        Ném `KeyError` nếu không thuộc library (trả về 404). Giới hạn độ dài theo
        `library_max_chars` — file quá dài bị cắt, báo `truncated` để UI biết.
        """
        if self._library_dir is None:
            raise KeyError("chưa khai data.library_dir")
        root = self._library_dir.resolve()
        target = (root / rel_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise KeyError(f"không có file {rel_path!r} trong library")
        code = target.read_text(encoding="utf-8", errors="replace")
        truncated = len(code) > self._library_max_chars
        if truncated:
            code = code[: self._library_max_chars]
        size = len(code)
        return {
            "path": rel_path,
            "name": target.name,
            "code": code,
            "n_lines": code.count("\n") + 1,
            "size": size,
            "truncated": truncated,
        }

    def scan_library(self, threshold: float | None = None) -> dict:
        """Quét all-vs-all bằng fingerprint MOSS, trả các cặp >= ngưỡng (giảm dần).

        Chỉ dùng B1 (MOSS) — nhanh và đúng thuật toán baseline. Cặp nào được bấm
        trên UI thì `analyze()` của 4 model chạy riêng cho cặp đó.
        """
        listing = self.library_files()
        files = listing["files"][: self._library_max_files]
        if not files:
            return {
                "n_files": 0, "threshold": threshold,
                "pairs": [], "duration_ms": 0.0, "mode": "moss",
            }
        if threshold is None:
            threshold = self._library_default_threshold
        threshold = float(threshold)
        threshold = max(0.0, min(1.0, threshold))

        moss_cfg = self.registry.config.get("models", {}).get(
            "B1", {}
        )
        k = int(moss_cfg.get("k", 20))
        w = int(moss_cfg.get("w", 5))

        from src.models.baseline.moss import FingerprintDoc, scan_pairwise

        t0 = time.perf_counter()
        docs: list[FingerprintDoc] = []
        for info in files:
            content = self.read_library_file(info["path"])
            docs.append(
                FingerprintDoc.from_code(
                    info["path"], content["code"], k=k, w=w
                )
            )
        pairs = scan_pairwise(docs, threshold)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "n_files": len(files),
            "threshold": threshold,
            "mode": "moss",
            "k": k,
            "w": w,
            "duration_ms": duration_ms,
            "pairs": pairs,
        }

    # ------------------------------------------------------- quét zip upload
    def upload_zip(self, data: bytes, filename: str) -> dict:
        """Nhận file .zip, bung các file code ra thư mục tạm, trả `upload_id`.

        Không tin tên file trong zip (chống `../../etc`): chỉ giữ basename,
        trùng tên thì thêm hậu tố. File không phải code / quá dài / lỗi decode
        thì BỎ QUA có ghi nhận (không fail cả upload). Quá giới hạn số file
        hoặc tổng bytes thì fail rõ ràng.
        """
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{filename} không phải file .zip hợp lệ") from exc
        entries: dict[str, str] = {}
        skipped: dict[str, int] = {"non_code": 0, "too_long": 0, "undecodable": 0,
                                   "dirs": 0}
        total_bytes = 0
        with zf:
            for info in zf.infolist():
                if info.is_dir():
                    skipped["dirs"] += 1
                    continue
                name = Path(info.filename).name
                if Path(name).suffix.lower() not in _LIBRARY_EXTS:
                    skipped["non_code"] += 1
                    continue
                try:
                    code = zf.read(info.filename).decode("utf-8")
                except (UnicodeDecodeError, RuntimeError):
                    skipped["undecodable"] += 1
                    continue
                total_bytes += len(code)
                if len(code) > self._scan_max_file_chars:
                    skipped["too_long"] += 1
                    continue
                if total_bytes > self._scan_max_bytes:
                    raise ValueError(
                        f"tổng dung lượng vượt {self._scan_max_bytes:,} bytes "
                        "— chia nhỏ zip rồi thử lại")
                base, i = name, 2
                while base in entries:
                    stem = Path(name).stem
                    base = f"{stem}_{i}{Path(name).suffix}"
                    i += 1
                entries[base] = code
        if len(entries) > self._scan_max_files:
            raise ValueError(
                f"{len(entries)} file, vượt giới hạn {self._scan_max_files} "
                "file/lần quét — tách zip nhỏ hơn rồi thử lại")
        if not entries:
            raise ValueError("zip không chứa file code nào "
                             f"(nhận: {sorted(_LIBRARY_EXTS)})")
        upload_id = uuid.uuid4().hex[:12]
        dest = self._upload_root / upload_id
        dest.mkdir(parents=True, exist_ok=True)
        for name, code in entries.items():
            (dest / name).write_text(code, encoding="utf-8")
        return {
            "upload_id": upload_id,
            "files": [{"name": n, "chars": len(c), "code": c}
                      for n, c in entries.items()],
            "skipped": skipped,
            "limits": {"max_files": self._scan_max_files,
                       "max_bytes": self._scan_max_bytes,
                       "max_file_chars": self._scan_max_file_chars},
        }

    def scan_upload(
        self,
        upload_id: str,
        model_ids: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Chấm all-pairs các file đã upload bằng model được chọn.

        `threshold=None` ⇒ dùng ngưỡng đóng băng từng model (đúng protocol báo
        cáo); cho số ⇒ áp chung cho mọi model. Cặp được giữ nếu CÓ ÍT NHẤT một
        model đạt ngưỡng, xếp giảm dần theo điểm cao nhất.
        """
        from src.demo.scorers.base import read_threshold

        if threshold is not None:
            threshold = max(0.0, min(1.0, float(threshold)))
        upload_dir = self._upload_root / upload_id
        if not upload_dir.is_dir():
            raise KeyError(f"không có upload {upload_id!r} — upload lại file .zip")
        files = {p.name: p.read_text(encoding="utf-8")
                 for p in sorted(upload_dir.glob("*")) if p.is_file()}
        scorers = self.registry.selected(model_ids)
        t0 = time.perf_counter()
        thr_of = {
            s.model_id: (threshold if threshold is not None
                         else read_threshold(s.results_dir))
            for s in scorers
        }
        names = sorted(files)
        pair_keys = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        raw: dict[str, dict[tuple[str, str], tuple[float | None, dict]]] = {}
        from src.demo.scorers.cpg_scorer import CpgScorer

        for s in scorers:
            if isinstance(s, CpgScorer) and pair_keys:
                raw[s.model_id] = s.score_all_pairs([(n, files[n]) for n in names])
            else:
                per = {}
                for a, b in pair_keys:
                    try:
                        r = s.score(files[a], files[b])
                        per[(a, b)] = (r.score, {})
                    except Exception as exc:  # noqa: BLE001 — model lỗi thì ô đó lỗi
                        per[(a, b)] = (None, {"error": f"{type(exc).__name__}: {exc}"[:200]})
                raw[s.model_id] = per
        pairs = []
        for a, b in pair_keys:
            cells = {}
            best: float | None = None
            for s in scorers:
                score, detail = raw[s.model_id][(a, b)]
                thr = thr_of[s.model_id]
                verdict = ("unknown" if (score is None or thr is None)
                           else ("clone" if score >= thr else "not_clone"))
                cells[s.model_id] = {
                    "score": None if score is None else round(float(score), 6),
                    "verdict": verdict,
                    "threshold": thr,
                    "error": detail.get("error"),
                }
                if score is not None and (best is None or score > best):
                    best = score
            keep = any(
                cells[s.model_id]["verdict"] == "clone" for s in scorers
            )
            if keep:
                pairs.append({"a": a, "b": b, "scores": cells,
                              "max_score": round(float(best), 6)})
        pairs.sort(key=lambda r: r["max_score"], reverse=True)
        return {
            "upload_id": upload_id,
            "n_files": len(files),
            "models": [s.model_id for s in scorers],
            "threshold_applied": thr_of,
            "n_pairs": len(pair_keys),
            "n_kept": len(pairs),
            "pairs": pairs,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ------------------------------------------------------------------ kiểm tra
    def _validate(self, code: str, field: str) -> None:
        if not code or not code.strip():
            raise ValueError(f"{field} rỗng")
        if len(code) > self.limits["max_code_chars"]:
            raise InputTooLarge(
                f"{field} dài {len(code):,} ký tự, vượt giới hạn "
                f"{self.limits['max_code_chars']:,}"
            )
        if code.count("\n") + 1 > self.limits["max_lines"]:
            raise InputTooLarge(
                f"{field} nhiều hơn {self.limits['max_lines']} dòng"
            )
