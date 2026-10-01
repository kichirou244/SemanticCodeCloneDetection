"""Chạy web demo: `python scripts/serve_demo.py`.

Cố ý dùng `reload=False` và `workers=1`: reload sẽ nạp lại toàn bộ 3 model trọng
126M tham số mỗi lần sửa file, mất hàng chọn giây mà không đổi gì trên UI (UI là
file tĩnh, sửa xong F5 là thấy).

Mặc định bind `127.0.0.1` — demo chỉ dành cho máy cá nhân, không mở ra LAN. Muốn
chia sẻ thì phải tự truyền `--host` và hiểu rằng không có auth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # chạy trực tiếp file này thì cần đường dẫn gốc
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.demo.registry import DEFAULT_CONFIG  # noqa: E402
from src.demo.server import build_service, create_app  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web demo phát hiện clone ngữ nghĩa")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="đường dẫn YAML")
    parser.add_argument("--host", default=None, help="mặc định lấy từ config (127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="mặc định lấy từ config")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"],
                        help="ghi đè thiết bị của config")
    parser.add_argument("--preload", default=None,
                        help="danh sách model nạp trước, phân tách bằng dấu phẩy; "
                             "rỗng = dùng `preload` trong config")
    parser.add_argument("--no-preload", action="store_true",
                        help="bỏ nạp trước, model nạp lười ở request đầu tiên")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    config = load_config(Path(args.config))
    if args.device:
        config.setdefault("runtime", {})["device"] = args.device
    service = build_service(args.config)
    if args.device:
        service.registry.config["runtime"]["device"] = args.device

    preload = [] if args.no_preload else (
        [m.strip() for m in args.preload.split(",") if m.strip()]
        if args.preload is not None
        else list(config.get("server", {}).get("preload", []))
    )

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or int(server_cfg.get("port", 8000))

    print(f"[demo] config      : {args.config}")
    print(f"[demo] device      : {service.registry.config.get('runtime', {}).get('device')}")
    print(f"[demo] models      : {', '.join(service.registry.order)}")
    print(f"[demo] preload     : {', '.join(preload) or '(khong)'}")
    if preload:
        for model_id, state in service.registry.preload(preload).items():
            print(f"[demo]   {model_id}: {state}")
    print(f"[demo] nen         : {len(service.registry.all())} model")
    print(f"[demo] mo web      : http://{host}:{port}")

    uvicorn.run(create_app(service), host=host, port=port, reload=False, workers=1)


if __name__ == "__main__":
    main()
