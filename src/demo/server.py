"""FastAPI — chỉ làm ba việc: validate input, gọi `DemoService`, trả JSON.

Cố ý không đặt logic ở đây. Nếu server tự căn chỉnh hay tự nạp model thì
`DemoService` mất ý nghĩa và không test được bằng `pytest` thuần (test đó không
dựng HTTP client). Giữ controller mỏng cũng nghĩa là `scripts/serve_demo.py` và
test dùng chung đúng một đường gọi.

Về preload: nạp model nặng (M2 1.44 GB) trong request đầu tiên sẽ treo trình
duyệt hàng chục giây. `POST /api/preload` tách việc đó ra để frontend hiện
skeleton và báo tiến độ từng model, thay vì quay một spinner không có thông tin.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.demo.registry import DEFAULT_CONFIG, ModelRegistry
from src.demo.samples import BUILTIN_SAMPLES, sample_to_dict
from src.demo.schemas import AnalyzeRequest, EvidenceRequest, PreloadRequest, ScanRequest
from src.demo.service import DemoService, InputTooLarge

WEB_DIR = Path(__file__).parent / "web"


def build_service(config_path: Path | str = DEFAULT_CONFIG, root: Path | None = None) -> DemoService:
    """Dựng `DemoService` từ YAML. Tách riêng để test không cần HTTP."""
    registry = ModelRegistry.from_yaml(config_path, root=root)
    demo_cfg = registry.config.get("demo", {})
    return DemoService(
        registry,
        max_code_chars=int(demo_cfg.get("limits", {}).get("max_code_chars", 20000)),
        max_lines=int(demo_cfg.get("limits", {}).get("max_lines", 400)),
        max_block_colors=int(registry.config.get("alignment", {}).get("n_block_colors", 6)),
    )


def create_app(service: DemoService | None = None, config_path: Path | str = DEFAULT_CONFIG,
               root: Path | None = None) -> FastAPI:
    svc = service or build_service(config_path, root=root)
    app = FastAPI(title="Semantic Code Clone Detection — Demo", version="1.0.0")
    app.state.service = svc

    @app.middleware("http")
    async def no_cache_static(request, call_next):
        """Demo chạy local: tắt cache trình duyệt cho file tĩnh.

        Không có middleware này, mỗi lần sửa JS/CSS là một lần vật lộn
        Ctrl+Shift+R (đã dính 3 lần: nút Tải lên chết im, tên file không hiện).
        Server local tải lại tức thì nên không-cache chẳng tốn gì.
        """
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/status")
    def api_status() -> dict:
        return svc.status()

    @app.get("/api/samples")
    def api_samples() -> list[dict]:
        return [sample_to_dict(s) for s in BUILTIN_SAMPLES]

    @app.get("/api/samples/real")
    def api_real_samples(split: str = "test", limit: int | None = None) -> dict:
        try:
            return {"split": split, "pairs": svc.real_pairs(split=split, limit=limit)}
        except OSError as exc:
            raise HTTPException(503, f"chưa có dữ liệu POJ-104: {exc}") from exc

    @app.get("/api/samples/real/{pair_id}")
    def api_real_sample(pair_id: str, split: str = "test") -> dict:
        try:
            return svc.load_real(pair_id, split=split)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/api/preload")
    def api_preload(req: PreloadRequest) -> dict:
        return svc.registry.preload(req.models)

    @app.post("/api/evidence")
    def api_evidence(req: EvidenceRequest) -> dict:
        """Evidence vị trí của một model cho một cặp (fetch riêng khi bấm card).

        Tách khỏi `/api/analyze` vì evidence M1/M2 tốn forward+backward riêng —
        gộp vào analyze là mỗi lần chấm gánh thêm giây + RAM transient.
        """
        try:
            return svc.evidence_for(req.model_id, req.code_a, req.code_b)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/uploads")
    async def api_upload(file: UploadFile = File(...)) -> dict:
        try:
            return svc.upload_zip(await file.read(), file.filename or "upload.zip")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/scan")
    def api_scan(req: ScanRequest) -> StreamingResponse:
        """Chấm all-pairs, stream tiến độ NDJSON (mỗi dòng một event).

        Mồi một `next()` trước để lỗi setup (upload không tồn tại, ngưỡng sai)
        trả đúng 404/400 thay vì sập giữa stream.
        """
        it = svc.scan_upload_events(req.upload_id, req.models, req.threshold)
        try:
            first = next(it)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        def gen():
            yield _ndjson(first[0], first[1])
            try:
                for kind, payload in it:
                    yield _ndjson(kind, payload)
            except Exception as exc:  # noqa: BLE001 — stream đang mở
                yield _ndjson("error", {"message": f"{type(exc).__name__}: {exc}"[:300]})

        return StreamingResponse(gen(), media_type="application/x-ndjson")



    @app.get("/api/library")
    def api_library() -> dict:
        """Danh sách file trong `data/library_dir` — cấu hình "database" để quét."""
        return svc.library_files()

    @app.get("/api/library/file")
    def api_library_file(path: str) -> dict:
        try:
            return svc.read_library_file(path)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f"không đọc được file: {exc}") from exc

    @app.post("/api/library/scan")
    def api_library_scan(threshold: float | None = None) -> dict:
        try:
            return svc.scan_library(threshold=threshold)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest) -> dict:
        try:
            return svc.analyze(
                code_a=req.code_a,
                code_b=req.code_b,
                model_ids=req.models,
                mode=req.mode,
                min_block_tokens=req.min_block_tokens,
            )
        except InputTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def _ndjson(kind: str, payload: dict) -> bytes:
    if kind == "result":
        return (json.dumps({"event": "result", "data": payload},
                           ensure_ascii=False) + "\n").encode("utf-8")
    if kind == "error":
        return (json.dumps({"event": "error", **payload},
                           ensure_ascii=False) + "\n").encode("utf-8")
    return (json.dumps({"event": "progress", **payload},
                       ensure_ascii=False) + "\n").encode("utf-8")
