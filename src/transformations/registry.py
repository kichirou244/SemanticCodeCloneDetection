"""Registry transformation bảo toàn ngữ nghĩa cho RQ4 (mở rộng ở phase sau).

INIT chỉ khóa tên + mô tả; engine cài đặt sau để không chặn core experiment.
"""

TRANSFORMATIONS = {
    "identifier_renaming": "Đổi tên biến/hàm, giữ ngữ nghĩa",
    "function_renaming": "Đổi tên hàm",
    "formatting": "Đổi format/khoảng trắng/chú thích",
    "dead_code_insertion": "Chèn code chết không ảnh hưởng hành vi",
    "statement_restructuring": "Tái cấu trúc câu lệnh tương đương",
    "control_flow_equiv": "Biến đổi control-flow tương đương",
    "loop_transformation": "for<->while và biến thể vòng lặp",
    "variable_substitution": "Thay biến bằng biểu thức tương đương",
}


def list_transformations() -> dict:
    return dict(TRANSFORMATIONS)
