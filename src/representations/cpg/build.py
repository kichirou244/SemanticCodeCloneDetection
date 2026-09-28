"""Xây dựng CPG = AST + CFG + Data Flow (M4, DEC-009 / DEC-024).

Trách nhiệm của module này chỉ là **trạng thái và điều phối** phần trích xuất:
- Backend C/C++ là **Joern CLI** (ngoài Python) → `scripts/joern/export_cpg.py`
  ghi JSON theo format của ta → `ingest.parse_cpg` → `CPGraph`.
- Backend Python (CPG-Python) đã bị **huỷ** theo DEC-019: scope chỉ còn
  POJ-104 C/C++, nên không giữ nhánh code chết.

Vì sao không gọi Joern từ đây: mỗi lần gọi là một JVM, và việc đó thuộc về
`scripts/build_cpg_export.py` (có lô, có đo giờ, có resume). Module này chỉ
trả lời "toolchain đã sẵn sàng chưa" để `preflight` chặn trước khi train.
"""

from __future__ import annotations

import shutil
import subprocess


def _version(cmd: list[str]) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *cmd[1:]], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout or out.stderr).strip().splitlines()
    return text[0] if text else ""


def cpg_status() -> dict:
    """Trạng thái toolchain CPG. Không tự cài gì — chỉ báo cáo + chỉ dẫn."""
    java = _version(["java", "-version"])
    joern = _version(["joern", "--version"])
    ready = java is not None and joern is not None
    return {
        "cpp_backend": "joern (CLI ngoài) -> scripts/joern/export_cpg.py -> ingest.parse_cpg",
        "python_backend": "removed (DEC-019: scope chỉ POJ-104 C/C++)",
        "java": java or "MISSING",
        "joern": joern or "MISSING",
        "ready": ready,
        "status": "ready" if ready else "blocked_on_joern",
        "where_to_install": "Kaggle notebook (notebooks/exp005_cpg_extract_kaggle.ipynb) — "
                            "không cài lên máy local: máy dự án không có Java, "
                            "còn Kaggle có sẵn JDK + quota CPU",
        "next_step": ("chạy scripts/build_cpg_export.py --limit 500 trên Kaggle để đo "
                      "ham/giay thật, rồi mới quyết định ngân sách giờ cho M4"),
    }
