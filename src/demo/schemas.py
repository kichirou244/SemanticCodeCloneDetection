"""DTO của API demo. Tách riêng khỏi `service.py` để hợp đồng HTTP đổi được
mà không đụng logic nghiên cứu bên dưới."""

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    code_a: str = Field(min_length=1, description="mã nguồn phía A")
    code_b: str = Field(min_length=1, description="mã nguồn phía B")
    models: list[str] | None = Field(
        default=None, description="rỗng = chấm tất cả model trong registry"
    )
    mode: str = Field(default="abstract", description="literal | abstract")
    min_block_tokens: int = Field(
        default=0, ge=0, le=64,
        description="0 = lấy theo configs/demo_web.yaml",
    )


class PreloadRequest(BaseModel):
    models: list[str] = Field(description="danh sách model_id cần nạp trước")


class ScanRequest(BaseModel):
    upload_id: str = Field(min_length=1, description="id từ POST /api/uploads")
    models: list[str] | None = Field(
        default=None, description="rỗng = chấm tất cả model trong registry"
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="None = dùng ngưỡng đóng băng từng model; cho số = áp chung",
    )
