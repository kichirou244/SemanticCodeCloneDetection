"""Lexer C++ cho web demo — sinh token kèm vị trí để căn chỉnh và tô màu.

Vì sao không dùng `code_tokenizer` của `src/models/baseline/tfidf.py`: hàm đó trả
về `list[str]` thuần, đủ cho TF-IDF nhưng **không giữ vị trí**, mà web demo cần
biết chính xác token nào nằm ở dòng nào để bôi đỏ xanh. Ở đây một lần duyệt
sinh ra cả hai thứ, kèm `kind` để tô màu cú pháp.

Hai khái niệm tách biệt, hay bị nhầm:

- `Token.text` — chuỗi gốc. Dùng để TÔ MÀU CÚ PHÁP.
- `Token.key(mode)` — khoá so khớp, phụ thuộc chế độ:
    * `literal`  → chính `text` (đối chiếu nguyên văn).
    * `abstract` → mọi identifier đổi thành `ID`, khoá C++ đổi thành lowercase.
      Đây là lớp trừu tượng gần với `src/representations/ast/extract.py` (lá chỉ
      còn ID/NUM/STR) — nhờ vậy một clone bị đổi tên biến vẫn hiện ra là "khớp",
      đúng thứ model semantic nhìn thấy và diff từ khoá nhìn thấy.

Cả hai chế độ đều bỏ qua comment và chuỗi literal khi so khớp, vì chúng hay khác
nhau giữa hai bản clone dù ngữ nghĩa giống nhau.

Cấu trúc: `lex_code` tách dòng trước rồi lex từng dòng, nên không có token nào
chứa newline ⇒ frontend vẽ 1 dòng = 1 danh sách token, không cần xử lý token đa
dòng. Comment khối `/* ... */` nhiều dòng được xử lý bằng cờ `in_block` mang qua
từng dòng.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Token nào bị loại khỏi danh sách so khớp. Comment và chuỗi literal giữ lại
#: trong payload để tô màu cú pháp, chỉ không tham gia căn chỉnh.
SKIP_KINDS = frozenset({"com", "str"})

_KEYWORDS = frozenset(
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char
    char16_t char32_t class compl const constexpr const_cast continue decltype
    default delete do double dynamic_cast else enum explicit export extern false
    float for friend goto if inline int long mutable namespace new noexcept not
    not_eq nullptr operator or or_eq private protected public register
    reinterpret_cast return short signed sizeof static static_assert static_cast
    struct switch template this throw true try typedef typeid typename union
    unsigned using virtual void volatile wchar_t while xor xor_eq
    """.split()
)

_PRIMITIVES = frozenset(
    "bool char double float int long short signed unsigned void size_t".split()
)

#: Nhóm tiền xử lý. Đứng đầu vì `#include <...>` không được cắt vỡ.
_PREPROCESSOR_RE = re.compile(r"^[ \t]*#")

#: Thứ tự khớp QUAN TRỌNG — nhóm nào khớp trước thắng. Comment và chuỗi phải đứng
#: trước `ident`/`num`/`op` để không bị cắt làm nhiều token.
_TOKEN_RE = re.compile(
    r"""
      (?P<com>//[^\n]*|/\*.*?\*/)
    | (?P<str>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
    | (?P<num>\.?[0-9](?:[eEpP][+-]|[A-Za-z0-9_.])*)
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<ws>[ \t]+)
    | (?P<op>>>>=|<<=|>>=|->\*|\.\.\.|<=>|::|->|\+\+|--|<<|>>|<=|>=|==|!=
              |&&|\|\||[-+*/%=<>!&|^~?:.,;(){}\[\]@#\\])
    | (?P<bad>.)
    """,
    re.VERBOSE,
)

_MULTI_CHAR_OPS = frozenset(
    {
        "->", "::", "&&", "||", "==", "!=", "<=", ">=", "++", "--", "<<", ">>",
        "+=", "-=", "*=", "/=", "%=", "->*", "...",
    }
)


@dataclass(frozen=True)
class Token:
    """Một token C++ trên một dòng, không chứa newline."""

    text: str
    kind: str
    line: int  # 0-based
    col: int  # 0-based, tính từ đầu dòng

    def key(self, mode: str = "literal") -> str:
        """Khoá so khớp theo chế độ đã chọn. Xem module docstring."""
        if mode == "abstract":
            return "ID" if self.kind == "id" else self.text.lower()
        return self.text


def classify(text: str) -> str:
    """Chuyển chuỗi token thô thành `kind` dùng cho tô màu cú pháp."""
    if not text:
        return "ws"
    if text[0].isspace():
        return "ws"
    if text.startswith("//") or text.startswith("/*"):
        return "com"
    if text[0] in "\"'":
        return "str"
    if text[0].isdigit() or (text[0] == "." and len(text) > 1 and text[1].isdigit()):
        return "num"
    if text[0].isalpha() or text[0] == "_":
        low = text.lower()
        if low in _KEYWORDS:
            return "kw"
        if low in _PRIMITIVES:
            return "type"
        return "id"
    return "op"


def lex_line(line_text: str, line: int, in_block: bool) -> tuple[list[Token], bool]:
    """Lex một dòng. Trả `(tokens, in_block_mới)`.

    `in_block` là trạng thái "đang nằm trong comment khối `/*`" do dòng trước bỏ
    lại — đây là lý do phải lex theo dòng mà không lex cả chuỗi.
    """
    if in_block:
        close = line_text.find("*/")
        if close == -1:
            return [Token(line_text, "com", line, 0)], True
        tokens = [Token(line_text[: close + 2], "com", line, 0)]
        tokens.extend(_lex_with_blocks(line_text[close + 2 :], line, close + 2))
        return tokens, False
    if _PREPROCESSOR_RE.match(line_text):
        return [Token(line_text, "pp", line, 0)], False
    return _lex_with_blocks(line_text, line, 0)


def _lex_with_blocks(text: str, line: int, col_offset: int) -> tuple[list[Token], bool]:
    """Lex một đoạn, tự phát hiện comment khối `/*` mở/chưa đóng ở ngay chỗ này.

    Khác `lex_line`: nhánh `in_block` chỉ tiếp quản khi dòng TRƯỚC đã để lại cờ —
    còn DÒNG NÀY vừa mở `/*` mà chưa đóng thì regex `/\*.*?\*/` bỏ qua, `/*` bị
    cắt vụn thành op. Hàm này tìm `/*` bằng máy quét nhẹ (bỏ qua `/*` trong chuỗi
    và trong comment `//`), nếu không có `*/` thì xử lý phần còn lại thành com và
    trả `in_block=True`.
    """
    open_pos = _find_block_open(text)
    if open_pos == -1:
        return lex_plain(text, line, col_offset), False
    close_rel = text.find("*/", open_pos + 2)
    if close_rel == -1:
        pre = lex_plain(text[:open_pos], line, col_offset)
        return pre + [Token(text[open_pos:], "com", line, col_offset + open_pos)], True
    pre = lex_plain(text[:open_pos], line, col_offset)
    com = Token(text[open_pos : close_rel + 2], "com", line, col_offset + open_pos)
    rest, _ = _lex_with_blocks(text[close_rel + 2 :], line, col_offset + close_rel + 2)
    return pre + [com] + rest, False


def _find_block_open(text: str) -> int:
    """Vị trí `/*` đầu tiên thật sự mở comment khối, hoặc -1 nếu không có.

    Bỏ qua `/*` nằm trong chuỗi literal hoặc trong comment `//` của cùng dòng.
    """
    i, n = 0, len(text)
    in_str: str | None = None
    while i < n - 1:
        ch = text[i]
        if in_str is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_str:
                in_str = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_str = ch
            i += 1
            continue
        if ch == "/" and text[i + 1] == "/":
            return -1
        if ch == "/" and text[i + 1] == "*":
            return i
        i += 1
    return -1


def lex_plain(text: str, line: int, col_offset: int) -> list[Token]:
    """Lex một đoạn không chứa comment khối đang mở."""
    return [
        Token(m.group(), classify(m.group()), line, col_offset + m.start())
        for m in _TOKEN_RE.finditer(text)
    ]


def lex_code(code: str) -> list[list[Token]]:
    """Lex cả chương trình, trả về danh sách theo dòng (0-based).

    Luôn trả về ít nhất một dòng rỗng để chuỗi rỗng vẫn render được.
    """
    lines = (code or "").splitlines() or [""]
    out: list[list[Token]] = []
    in_block = False
    for i, text in enumerate(lines):
        tokens, in_block = lex_line(text, i, in_block)
        out.append(tokens)
    return out
