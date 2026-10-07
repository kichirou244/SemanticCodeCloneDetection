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

#: Số cặp mỗi chunk khi quét: đủ lớn để scorer nặng forward gộp hiệu quả,
#: đủ nhỏ để tiến độ báo đều (1900 cặp → ~240 lần cập nhật).
SCAN_CHUNK_PAIRS = 8

#: Loại file được quét trong chế độ "database". Không nhận .txt — tránh nhiễu.
_LIBRARY_EXTS = frozenset({
    ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".cu", ".ino",
})


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _spans_to_indices(lines: list, code: str, spans) -> list[int]:
    """Map spans ký tự (code gốc) → chỉ số token phẳng (đúng thứ tự payload).

    Tính ở backend vì chỉ backend có đồng thời code + token đã lex — frontend
    đoán offset (nhất là `\r\n`) sẽ lệch. Token nào giao spans thì dính.
    """
    starts: list[int] = []
    pos = 0
    for line in code.split("\n"):
        starts.append(pos)
        pos += len(line) + 1
    out: list[int] = []
    flat = 0
    for li, tokens in enumerate(lines):
        base = starts[li] if li < len(starts) else None
        for tok in tokens:
            if base is not None:
                s = base + tok.col
                e = s + len(tok.text)
                if any(ss < e and s < se for ss, se in spans):
                    out.append(flat)
            flat += 1
    return out


def _attach_token_indices(ev, lines_a, lines_b, code_a, code_b):
    """Gắn `tok_a/tok_b` (chỉ số token để frontend tô) vào evidence spans."""
    if not ev:
        return None
    return {
        "kind": ev.get("kind"),
        "note": ev.get("note"),
        "n_shared": ev.get("n_shared"),
        "spans_a": ev.get("spans_a", []),
        "spans_b": ev.get("spans_b", []),
        "tok_a": _spans_to_indices(lines_a, code_a, ev.get("spans_a", [])),
        "tok_b": _spans_to_indices(lines_b, code_b, ev.get("spans_b", [])),
    }


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
        # KHÔNG tính evidence ở đây: evidence M1/M2 tốn forward+backward riêng,
        # nhét vào analyze là mỗi lần chấm gánh thêm giây + RAM transient
        # (đã OOM chết server một lần). Evidence fetch riêng khi bấm card.
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

    def evidence_for(self, model_id: str, code_a: str, code_b: str) -> dict:
        """Evidence vị trí của một model cho một cặp (fetch riêng theo click).

        Trả `{"model_id", "evidence": {...} | None}` — None nghĩa là model này
        chưa có evidence riêng (M4: cần exporter thêm vị trí dòng), UI giữ
        nguyên highlight chung và báo rõ, thay vì đoán.
        """
        from src.demo.alignment.lexer import lex_code

        scorer = self.registry.get(model_id)
        lines_a, lines_b = lex_code(code_a), lex_code(code_b)
        try:
            ev = scorer.evidence(code_a, code_b)
        except Exception as exc:  # noqa: BLE001 — lỗi evidence không sập request
            return {"model_id": model_id, "evidence": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200]}
        return {"model_id": model_id,
                "evidence": _attach_token_indices(
                    ev, lines_a, lines_b, code_a, code_b)}

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
        """Quét all-vs-all bằng MOSS, trả các cặp >= ngưỡng (giảm dần).

        Chỉ dùng B1. B1 gọi MOSS thật (moss.stanford.edu) với MỘT lần nộp cho cả
        thư mục (`report_url` trong kết quả); thiếu MOSS_USER_ID thì tự fallback
        winnowing local (k=20, w=5). Cặp nào được bấm trên UI thì `analyze()` của
        4 model chạy riêng cho cặp đó.
        """
        from src.demo.scorers.moss_scorer import MossScorer

        listing = self.library_files()
        files = listing["files"][: self._library_max_files]
        if not files:
            return {
                "n_files": 0, "threshold": threshold, "mode": "moss",
                "pairs": [], "duration_ms": 0.0,
            }
        if threshold is None:
            threshold = self._library_default_threshold
        threshold = max(0.0, min(float(threshold), 1.0))

        moss = self.registry.get("B1")
        if not isinstance(moss, MossScorer):  # config B1 sai kind — fail rõ ràng
            raise RuntimeError("B1 trong registry không phải MossScorer")

        t0 = time.perf_counter()
        items = [(info["path"], self.read_library_file(info["path"])["code"])
                 for info in files]
        raw = moss.score_all_pairs(items)
        pairs = sorted(
            ({"a": a, "b": b, "similarity": round(float(score), 6)}
             for (a, b), (score, _detail) in raw.items()
             if score is not None and score >= threshold),
            key=lambda r: r["similarity"], reverse=True,
        )
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        out = {
            "n_files": len(files),
            "threshold": threshold,
            "mode": "moss",
            "duration_ms": duration_ms,
            "pairs": pairs,
        }
        if moss.mode == "moss-service":
            # link report MOSS thật để mở kết quả gốc ở Stanford
            first_url = next((d.get("report_url") for (_a, _b), (s, d) in
                              raw.items() if d.get("report_url")), None)
            out["report_url"] = first_url
            out["cache_dir"] = str(moss.cache_dir) if moss.cache_dir else None
        return out

    # ------------------------------------------------------- quét zip upload
    def upload_zip(self, data: bytes, filename: str) -> dict:
        """Nhận file .zip, bung các file code ra thư mục tạm, trả `upload_id`.

        Không tin tên file trong zip (chống `../../etc`): chỉ giữ basename,
        trùng tên thì thêm hậu tố. File không phải code / quá dài / lỗi decode
        thì BỎ QUA có ghi nhận (không fail cả upload). Không giới hạn số file —
        chỉ chặn tổng bytes (zip bomb); quét lâu thì tiến độ báo live từng cặp.
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
            "limits": {"max_bytes": self._scan_max_bytes,
                       "max_file_chars": self._scan_max_file_chars},
        }

    def scan_upload_events(self, upload_id, model_ids=None, threshold=None):
        """Generator quét all-pairs: yield `("progress", {...})` từng chặng rồi
        `("result", {...})` cuối cùng.

        Setup (đọc upload, chọn model) nằm TRƯỚC yield đầu nên lỗi `KeyError` /
        `ValueError` ném ngay — endpoint mồi một `next()` để trả đúng 404/400.
        Lỗi giữa chừng (hiếm) thành event `"error"` chứ không sập stream.
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
        total = len(pair_keys)
        from src.demo.scorers.cpg_scorer import CpgScorer
        from src.demo.scorers.moss_scorer import MossScorer

        # M4 batch trước: 1 lần Joern cho mọi file (không phải N² lần JVM).
        # B1 cũng batch: 1 lần nộp MOSS cho cả zip (không phải N² lần upload).
        raw: dict[str, dict[tuple[str, str], tuple[float | None, dict]]] = {}
        for s in scorers:
            raw[s.model_id] = {}
            if isinstance(s, (CpgScorer, MossScorer)) and pair_keys:
                raw[s.model_id] = s.score_all_pairs([(n, files[n]) for n in names])
        pairs = []
        done = 0
        try:
            # Chunk-outer: mỗi chunk chấm xong cho MỌI model rồi mới dựng hàng,
            # nên tiến độ rơi đúng vào phase chấm chậm (M1), không phải lúc cuối.
            for ci in range(0, len(pair_keys), SCAN_CHUNK_PAIRS):
                chunk = pair_keys[ci : ci + SCAN_CHUNK_PAIRS]
                for s in scorers:
                    if isinstance(s, (CpgScorer, MossScorer)):
                        continue  # đã batch đủ ở trên
                    try:
                        got = s.score_batch([(files[a], files[b]) for a, b in chunk])
                    except Exception as exc:  # noqa: BLE001 — hỏng cả chunk thì ô đó lỗi
                        got = [(None, {"error": f"{type(exc).__name__}: {exc}"[:200]})
                               for _ in chunk]
                    for (a, b), (score, detail) in zip(chunk, got):
                        raw[s.model_id][(a, b)] = (score, detail)
                for a, b in chunk:
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
                            "report_url": detail.get("report_url"),
                        }
                        if score is not None and (best is None or score > best):
                            best = score
                    keep = any(
                        cells[s.model_id]["verdict"] == "clone" for s in scorers
                    )
                    if keep:
                        pairs.append({"a": a, "b": b, "scores": cells,
                                      "max_score": round(float(best), 6)})
                done += len(chunk)
                yield ("progress", {"done": done, "total": total,
                                    "kept": len(pairs)})
        except Exception as exc:  # noqa: BLE001 — stream đang mở, chỉ còn cách event lỗi
            yield ("error", {"message": f"{type(exc).__name__}: {exc}"[:300]})
            return
        pairs.sort(key=lambda r: r["max_score"], reverse=True)
        yield ("result", {
            "upload_id": upload_id,
            "n_files": len(files),
            "models": [s.model_id for s in scorers],
            "threshold_applied": thr_of,
            "n_pairs": len(pair_keys),
            "n_kept": len(pairs),
            "pairs": pairs,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        })

    def scan_upload(
        self,
        upload_id: str,
        model_ids: list[str] | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Bản đồng bộ của `scan_upload_events` (giữ cho test + tương thích):
        gom tới event `result`, gặp `error` thì ném."""
        for kind, payload in self.scan_upload_events(upload_id, model_ids, threshold):
            if kind == "result":
                return payload
            if kind == "error":
                raise RuntimeError(payload["message"])
        raise RuntimeError("unreachable")

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
