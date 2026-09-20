"""Xây dựng CPG = AST + CFG + Data Flow (abstraction, Step 14).

C/C++: Joern (CLI ngoài) → đồ thị; Python: xây dựng từ AST (ast module) +
CFG suy từ cấu trúc điều khiển. Implement đầy đủ ở phase sau khi
feasibility (reports/) được người dùng xác nhận toolchain.
"""


def cpg_status() -> dict:
    return {
        "cpp_backend": "joern (chưa kiểm tra CLI — xem feasibility)",
        "python_backend": "ast + CFG tự xây (chưa implement)",
        "status": "feasibility_pending",
    }
