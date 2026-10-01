"""Unit test cho lexer C++ của demo — các trường hợp hay làm lệch hàng.

Bộ test này chạy không cần model hay FastAPI: chỉ cần `src.demo.alignment.lexer`.
"""

from __future__ import annotations

import pytest

from src.demo.alignment.lexer import Token, lex_code


def flat(code: str) -> list[Token]:
    return [t for tokens in lex_code(code) for t in tokens]


def kinds_of(code: str) -> list[str]:
    return [t.kind for t in flat(code)]


def texts_of(code: str) -> list[str]:
    return [t.text for t in flat(code)]


def test_simple_program_kinds():
    kinds = kinds_of("int main() { return 7; }")
    assert "type" in kinds      # int
    assert "id" in kinds        # main
    assert "kw" in kinds        # return
    assert "num" in kinds       # 7
    assert "op" in kinds        # ( ) { } ;
    assert kinds.count("kw") == 1
    assert kinds.count("type") == 1


def test_abstract_key_normalizes_identifiers():
    assert Token("tmp", "id", 0, 0).key("abstract") == "ID"
    assert Token("tmp", "id", 0, 0).key("literal") == "tmp"
    # keyword và primitive được lowercase hoá, identifier thì thành ID
    assert Token("Return", "kw", 0, 0).key("abstract") == "return"
    assert Token("int", "type", 0, 0).key("abstract") == "int"
    assert Token("++", "op", 0, 0).key("abstract") == "++"


def test_block_comment_spanning_lines_does_not_desync():
    code = "int a = 1; /* multi\n   line comment */ int b = 2;"
    tokens = lex_code(code)
    kinds = [t.kind for t in tokens[0]] + [t.kind for t in tokens[1]]
    assert "com" in kinds
    # sau khi comment đóng giữa dòng 2, token `b` thật vẫn xuất hiện
    assert any(t.kind == "id" and t.text == "b" for t in tokens[1])
    # thứ tự dòng không bị lệch: a dòng 1, b dòng 2
    assert any(t.text == "a" for t in tokens[0])


def test_block_comment_starting_mid_line_without_close():
    # dòng ĐẦU mở `/*` mà không đóng: phần còn lại phải là com, dòng sau tiếp tục
    code = "int x = 5; /* never closes\nstill commented\nafter */ int y = 3;"
    tokens = lex_code(code)
    assert any(t.kind == "com" and t.text.startswith("/*") for t in tokens[0])
    assert all(t.kind == "com" for t in tokens[1])
    # dòng thứ ba: sau `*/` vẫn còn `int y = 3;`
    assert any(t.kind == "id" and t.text == "y" for t in tokens[2])


def test_block_open_inside_string_is_not_comment():
    code = 'printf("a /* b */"); /* real */ int z = 1;'
    kinds = kinds_of(code)
    com_count = kinds.count("com")
    assert com_count == 1  # chỉ một comment thật
    tokens = flat(code)
    st = [t for t in tokens if t.kind == "str"]
    assert st and st[0].text == '"a /* b */"'


def test_slash_slash_after_block_open():
    # `//` xuất hiện trước `/*` → mọi thứ sau `//` là comment dòng, không tách `/*`
    code = "int a; // note /* fake */"
    tokens = lex_code(code)[0]
    comments = [t for t in tokens if t.kind == "com"]
    assert len(comments) == 1 and comments[0].text == "// note /* fake */"
    assert not any(t.text == "/*" for t in tokens)


def test_string_literal_is_single_token():
    tokens = flat('printf("hello /* not comment */ world");')
    strings = [t for t in tokens if t.kind == "str"]
    assert strings and strings[0].text == '"hello /* not comment */ world"'
    assert all(t.kind != "com" for t in tokens)


def test_empty_and_whitespace_only():
    codes = lex_code("")
    assert len(codes) == 1 and codes[0] == []
    assert kinds_of("   \n\t") == ["ws", "ws"]


def test_token_has_line_and_col():
    first = lex_code("int x;")[0][0]
    assert isinstance(first, Token)
    assert first.line == 0
    assert first.col == 0
    assert first.text == "int"


def test_increment_is_single_op_token():
    # `++` giữ nguyên một token op (không tách thành `+` `+`)
    texts = texts_of("i++;")
    assert texts.count("++") == 1
    assert kinds_of("i++;").count("op") == 2  # ++ và ;


def test_preprocessor_line_kept_whole():
    code = "#define MAX(a, b) ((a) > (b) ? (a) : (b))\nint x = MAX(3, 5);"
    tokens = lex_code(code)
    assert tokens[0] == [Token(code.splitlines()[0], "pp", 0, 0)]


@pytest.mark.parametrize("code", [
    "struct P { int x; int y; };",
    "typedef unsigned long long ull;",
    "for (int i = 0; i < n; i++) { s += i; }",
    'char s[] = "hi\\\\n"; //x',
])
def test_lexer_covers_all_source_chars(code: str):
    # nối lại mọi token (trừ ws) phải phủ đúng mọi ký tự nguồn không phải khoảng trắng.
    # Lưu ý: comment/chuỗi có chứa space nội bộ thì invariant này không còn đúng —
    # các trường hợp trên được chọn không có space bên trong str/com.
    all_tokens = flat(code)
    joined = "".join(t.text for t in all_tokens if t.kind != "ws")
    source = "".join(code.split())
    assert joined == source