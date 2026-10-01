"""Căn chỉnh hai đoạn code và dựng payload sẵn để frontend vẽ.

Cơ chế: đưa **khoá so khớp** của token hai phía vào `difflib.SequenceMatcher` rồi
đọc `get_opcodes()`. Mỗi opcode `equal` là một **khối khớp**. Khối đó được đánh số,
và cùng một số ở cả hai phía là thứ cho phép frontend tô cùng một màu — tức "các
kiểu các đoạn khớp".

Một chi tiết dễ sai: `SequenceMatcher` so sánh **phần tử** bằng `==`, nên phần
tử đưa vào phải **chỉ là khoá khớp** (`"ID"`, `"return"`, `";"`). Nếu nhét kèm
`(line, col)` vào thì không token nào bằng nhau nữa và luật khớp biến mất. Vì vậy
`align()` giữ hai mảng song song: `keys` để so khớp, `positions` để ánh xạ ngược
về tọa độ để tô màu.

`difflib` tìm đường khớp tối ưu theo tổng độ dài trong O(n·m) (với
`autojunk=False`). Ở đây n chỉ vài nghìn token (đã chặn ở `max_lines` của config)
nên không đáng kể; nếu sau này cho phép file lớn thì đây là chỗ phải đổi sang
token-LCS theo hash bucket, không phải chỗ khác.

Hai điều kiện lọc quyết định chất lượng hình ảnh:

- `min_block_tokens` — khối chỉ gồm `;`/`)`/`{` rải rác thì bôi màu ra rác.
- `SKIP_KINDS` — comment và chuỗi literal loại khỏi so khớp (xem `lexer`).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from src.demo.alignment.lexer import SKIP_KINDS, Token

#: `(line, col, col_end)` — vị trí của một token trong danh sách phẳng.
Pos = tuple[int, int, int]


@dataclass(frozen=True)
class MatchBlock:
    """Một đoạn khớp, nhìn thấy đồng thời ở cả hai phía."""

    id: int
    a_range: tuple[int, int]  # [start, end) trong `keys_a`
    b_range: tuple[int, int]
    a_lines: tuple[int, int]  # [start, end) — dòng 0-based
    b_lines: tuple[int, int]
    a_n_tokens: int
    b_n_tokens: int

    @property
    def size(self) -> int:
        return min(self.a_n_tokens, self.b_n_tokens)


@dataclass
class AlignmentResult:
    """Kết quả căn chỉnh: khối khớp + thống kê phủ khớp + payload render."""

    mode: str
    blocks: list[MatchBlock] = field(default_factory=list)
    n_tokens_a: int = 0
    n_tokens_b: int = 0
    n_matched_a: int = 0
    n_matched_b: int = 0

    @property
    def coverage_a(self) -> float:
        return self.n_matched_a / self.n_tokens_a if self.n_tokens_a else 0.0

    @property
    def coverage_b(self) -> float:
        return self.n_matched_b / self.n_tokens_b if self.n_tokens_b else 0.0

    def to_payload(self, lines_a: list[list[Token]], lines_b: list[list[Token]]) -> dict:
        """Dựng payload cho frontend: mỗi token mang `kind` + `b` (id khối, -1 nếu lệch).

        Frontend chỉ việc map `kind` → class cú pháp và `b` → class màu khối; nó
        không cần biết thuật toán căn chỉnh là gì.
        """
        return {
            "mode": self.mode,
            "stats": {
                "n_tokens_a": self.n_tokens_a,
                "n_tokens_b": self.n_tokens_b,
                "n_matched_a": self.n_matched_a,
                "n_matched_b": self.n_matched_b,
                "coverage_a": round(self.coverage_a, 4),
                "coverage_b": round(self.coverage_b, 4),
                "n_blocks": len(self.blocks),
            },
            "blocks": [
                {
                    "id": blk.id,
                    "a": list(blk.a_lines),
                    "b": list(blk.b_lines),
                    "size": blk.size,
                }
                for blk in self.blocks
            ],
            "a": _dump_side(lines_a, _paint(lines_a, self.blocks, "a")),
            "b": _dump_side(lines_b, _paint(lines_b, self.blocks, "b")),
        }


def align(
    lines_a: list[list[Token]],
    lines_b: list[list[Token]],
    mode: str = "abstract",
    min_block_tokens: int = 2,
) -> AlignmentResult:
    """Căn chỉnh hai phía theo `mode`, trả `AlignmentResult`."""
    keys_a, keys_b = _keys(lines_a, mode), _keys(lines_b, mode)
    result = AlignmentResult(mode=mode, n_tokens_a=len(keys_a), n_tokens_b=len(keys_b))
    if not keys_a or not keys_b:
        return result

    pos_a, pos_b = _positions(lines_a), _positions(lines_b)
    matcher = difflib.SequenceMatcher(None, keys_a, keys_b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal" or (i2 - i1) < min_block_tokens:
            continue
        result.blocks.append(
            MatchBlock(
                id=len(result.blocks),
                a_range=(i1, i2),
                b_range=(j1, j2),
                a_lines=_line_span(pos_a[i1:i2]),
                b_lines=_line_span(pos_b[j1:j2]),
                a_n_tokens=i2 - i1,
                b_n_tokens=j2 - j1,
            )
        )
        result.n_matched_a += i2 - i1
        result.n_matched_b += j2 - j1
    return result


def _keys(lines: list[list[Token]], mode: str) -> list[str]:
    """Dàn khoá so khớp của các token tham gia, theo thứ tự đọc.

    Thứ tự phải khớp chính xác với `_positions` — cùng một tiêu chí lọc
    (`SKIP_KINDS`), cùng một vòng duyệt.
    """
    return [
        tok.key(mode)
        for tokens in lines
        for tok in tokens
        if tok.kind not in SKIP_KINDS
    ]


def _positions(lines: list[list[Token]]) -> list[Pos]:
    """Vị trí của các token tham gia so khớp, đúng thứ tự với khoá `keys`.

    Tách riêng `_positions` khỏi `_flatten` vì nó không phụ thuộc `mode`: cùng một
    tọa độ dùng được cho cả hai chế độ, nên không cần dàn lại khoá chỉ để lấy vị trí.
    """
    out: list[Pos] = []
    for tokens in lines:
        for tok in tokens:
            if tok.kind in SKIP_KINDS:
                continue
            out.append((tok.line, tok.col, tok.col + len(tok.text)))
    return out


def _line_span(positions: list[Pos]) -> tuple[int, int]:
    """Từ vị trí các token của một khối, suy ra khoảng dòng `[start, end)` mà chúng chiếm.

    Khối có thể bắt đầu giữa dòng này và kết thúc giữa dòng kia, nên số token thật
    phải giữ riêng, không suy ra từ khoảng dòng.
    """
    return positions[0][0], positions[-1][0] + 1


def _paint(
    lines: list[list[Token]], blocks: list[MatchBlock], side: str
) -> dict[int, dict[int, int]]:
    """Gán id khối cho từng token: trả `{line: {col: block_id}}`.

    Duyệt trực tiếp theo khoảng chỉ số token của khối, nên chính xác tuyệt đối —
    token nằm giữa dòng vẫn được bôi đúng. Vì các khối từ `SequenceMatcher` là
    rời nhau trên trục token, một cột không thể thuộc hai khối, nên ghi đè cũng vô
    hại.
    """
    positions = _positions(lines)
    grid: dict[int, dict[int, int]] = {}
    for blk in blocks:
        lo, hi = blk.a_range if side == "a" else blk.b_range
        for line, col, _ in positions[lo:hi]:
            grid.setdefault(line, {})[col] = blk.id
    return grid


def _dump_side(lines: list[list[Token]], grid: dict[int, dict[int, int]]) -> dict:
    """Chuyển token + lưới tô màu thành payload gọn cho frontend.

    `b = -1` nghĩa là token không thuộc khối khớp nào ⇒ frontend tô đỏ. Đây là
    thông tin **về mặt hình thức**, không phải kết luận về ngữ nghĩa: một clone
    đổi tên biến sẽ có nhiều `b = -1` ở chế độ `literal` nhưng gần như hết ở chế
    độ `abstract`.
    """
    return {
        "lines": [
            {
                "n": idx + 1,
                "tokens": [
                    {"t": t.text, "k": t.kind, "b": grid.get(idx, {}).get(t.col, -1)}
                    for t in tokens
                ],
            }
            for idx, tokens in enumerate(lines)
        ]
    }
