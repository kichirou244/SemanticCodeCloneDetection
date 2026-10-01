"""Lớp căn chỉnh hai đoạn code — lexer + so khớp token + payload cho frontend."""

from src.demo.alignment.aligner import AlignmentResult, MatchBlock, align
from src.demo.alignment.lexer import SKIP_KINDS, Token, lex_code

__all__ = ["SKIP_KINDS", "AlignmentResult", "MatchBlock", "Token", "align", "lex_code"]
