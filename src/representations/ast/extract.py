"""Trích xuất AST bằng tree-sitter (abstraction theo language, Step 14)."""

SUPPORTED = {"cpp": "tree-sitter-cpp", "python": "tree-sitter-python"}


def check_toolchain() -> dict:
    """Kiểm tra parser khả dụng mà không cần model nặng. Dùng cho feasibility."""
    report: dict = {"tree_sitter": False, "grammars": {}}
    try:
        import tree_sitter  # noqa: F401

        report["tree_sitter"] = True
    except ImportError:
        report["tree_sitter"] = False
    for lang, pkg in SUPPORTED.items():
        try:
            __import__(pkg.replace("-", "_"))
            report["grammars"][lang] = True
        except ImportError:
            report["grammars"][lang] = False
    return report
