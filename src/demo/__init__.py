"""Web demo phát hiện clone — chỉ đọc artifact, không train, không ghi `results/`.

Bốn tầng, mỗi tầng một trách nhiệm:

| Tầng | Module | Trách nhiệm |
| :--- | :--- | :--- |
| Căn chỉnh | `alignment/` | Lexer C++ + so khớp token → payload tô màu |
| Chấm điểm | `scorers/` | Bọc mỗi model thành một `Scorer` thống nhất |
| Điều phối | `registry.py`, `service.py` | Nạp lười model, ghép kết quả, áp ngưỡng |
| Giao diện | `server.py`, `web/` | FastAPI + frontend tĩnh |

Export nặng (create_app → fastapi) để **lười**: `tests/test_demo_lexer.py` chỉ
import `src.demo.alignment`, phải chạy được ngay cả khi chưa cài fastapi. Nếu
`__init__` import thẳng server ở module level thì test đó cũng phải đòi fastapi.
"""

from __future__ import annotations

_LAZY_EXPORTS = {"DemoService": "src.demo.service", "create_app": "src.demo.server"}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} không có {name!r}")


__all__ = ["DemoService", "create_app"]