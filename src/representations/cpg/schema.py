"""Vocabulary CPG cho M4 — **hợp đồng** giữa Joern (ngoài) và mô hình (trong).

Vì sao tách riêng khỏi extractor: Joern đổi tên node type giữa các version, còn
vocabulary đầu vào của mô hình phải **đóng băng** (DEC-009). Đổi bảng này = đổi
feature = mọi checkpoint đã train trở nên vô dùng, nên mọi thay đổi phải kèm
decision log.

Chỉ giữ nhãn thực sự xuất hiện với C/C++ trên POJ-104. Nhãn lạ (kể cả do Joern đổi
tên) rơi về `OTHER` hoặc bị bỏ — và **luôn được đếm lại** trong stats: im lặng
nuốt nhãn là cách hay nhất để train xong mà không biết mình đang mất thông tin gì.
"""

from __future__ import annotations

import math

# Nhãn node theo CPG schema của Joern, rút gọn còn phần dùng được cho C/C++.
NODE_LABELS: tuple[str, ...] = (
    "METHOD",            # đỉnh gốc của đồ thị một hàm
    "PARAMETER",         # tham số hàm
    "LOCAL",             # biến cục bộ
    "FIELD",             # trường của struct/class
    "CALL",              # lời gọi hàm
    "IDENTIFIER",        # tên biến/hàm được tham chiếu
    "LITERAL",           # hằng số/chuỗi
    "OPERATOR",          # toán tử
    "CONTROL_STRUCTURE", # if/for/while/...
    "JUMP_TARGET",       # nhãn nhảy (goto/label)
    "BLOCK",             # khối lệnh
    "TYPE_DECL",         # khai báo kiểu
    "OTHER",             # nhãn lạ — xem node_type_id
)

# Nhãn cạnh. Tách AST (cây cú pháp) / CFG (luồng điều khiển) / data-flow (REACHING_DEF)
# là bản chất của CPG: bỏ nhóm nào thì mô hình không còn là CPG-GNN nữa.
EDGE_LABELS: tuple[str, ...] = (
    "AST",
    "CFG",
    "CALL",
    "REF",
    "ARGUMENT",
    "TARG",
    "PARAMETER_OUT",
    "REACHING_DEF",
    "EVAL_ORDER",
)

# `OTHER` cuối danh sách node, nhãn lạ của edge trả -1 (bỏ) — xem edge_type_id.
NODE_OTHER = NODE_LABELS[-1]
N_NODE_TYPES = len(NODE_LABELS)
N_EDGE_TYPES = len(EDGE_LABELS)

# Số đối số của lời gọi, chuẩn hoá log1p: tín hiệu ngữ nghĩa mà node type không
# nắm được (CALL tới hàm 0 tham số khác CALL tới hàm 5 tham số). `log1p` + chia 5
# giữ giá trị trong [0, ~0,55] để không át tín hiệu của embedding.
FEATURE_DIM = 1
_ARG_COUNT_SCALE = 5.0

_NODE_TO_ID = {label: i for i, label in enumerate(NODE_LABELS)}
_EDGE_TO_ID = {label: i for i, label in enumerate(EDGE_LABELS)}


def node_type_id(label: str) -> int:
    """Id của nhãn node; nhãn lạ → `OTHER` (không ném lỗi, vì Joern sinh nhãn
    theo version và ném lỗi ở giữa lúc parse 62k code là cách mất cả phiên)."""
    return _NODE_TO_ID.get(str(label).upper(), _NODE_TO_ID[NODE_OTHER])


def edge_type_id(label: str) -> int:
    """Id của nhãn cạnh; nhãn lạ → -1 nghĩa là **bỏ cạnh**.

    Bỏ (chứ không gộp về nhãn nào) vì cạnh lạ thường là quan hệ *không thuộc*
    schema ta dùng (ví dụ `INCLUDES` giữa file) — giữ lại sẽ khiến mô hình học
    quan hệ không liên quan tới clone.
    """
    return _EDGE_TO_ID.get(str(label).upper(), -1)


def arg_count_feature(n_args: int) -> float:
    """log1p(n_args)/5, chặn trên 1.0. `n_args < 0` (thiếu dữ liệu) → 0.0."""
    if n_args is None or n_args < 0:
        return 0.0
    return min(math.log1p(float(n_args)) / _ARG_COUNT_SCALE, 1.0)
