"""Trích xuất AST bằng tree-sitter + abstraction theo ASTNN (M3).

Tham chiếu: Zhang et al. (2019), "AstNN: AST Neural Network for Code Clone
Detection".

Nguyên tắc abstraction (mục 3.1 bài báo — bỏ identifiers vì biến thể vô hạn):
  - Node lá (token) chỉ giữ 3 lớp: `ID` / `NUM` / `STR`.
  - Node trong chỉ giữ **loại statement** của nó; mọi token bên trong bị loại
    bỏ (keywords, operators đều drop → cây chỉ còn skeleton).
  - Node không phải statement (expression, argument_list, ...) bị **flatten**:
    bỏ node đó, giữ lại các lá bên trong và gắn thẳng vào statement cha.

Vector đầu vào của 1 statement = `W_x · mean(emb(lá thuộc cây con nó))`. Gom
lá thành 1 vector là chính là chỗ tránh nổ OOM so với giữ từng token làm node
riêng.

Output là **numpy arrays phẳng** (không phải object tree) để collation nhanh:
biết `n = len(label)` là biết ngay kích thước, không cần `max()` trên list.
"""

from __future__ import annotations

import numpy as np
from tree_sitter import Language, Node, Parser

SUPPORTED = {
    "cpp": "tree-sitter-cpp",
    "python": "tree-sitter-python",
    "java": "tree-sitter-java",
}

# --- node lá: 3 lớp abstraction ------------------------------------------
LEAF_ID, LEAF_NUM, LEAF_STR = 0, 1, 2
LEAF_LABELS = ("ID", "NUM", "STR")

# --- node trong: loại statement ------------------------------------------
STMT_TYPES_CPP = (
    "translation_unit", "function_definition", "declaration", "compound_statement",
    "if_statement", "for_statement", "while_statement", "do_statement",
    "switch_statement", "return_statement", "expression_statement",
    "for_range_loop", "catch_clause", "condition_clause", "break_statement",
    "continue_statement", "labeled_statement", "try_statement",
    "field_declaration", "struct_specifier", "class_specifier",
    "enum_specifier", "template_declaration", "namespace_definition",
    "lambda_expression", "array_declarator", "init_declarator",
)
STMT_TYPES_JAVA = (
    "program", "method_declaration", "constructor_declaration", "class_declaration",
    "interface_declaration", "enum_declaration", "record_declaration",
    "block", "local_variable_declaration", "enhanced_for_statement",
    "throw_statement", "try_with_resources_statement", "finally_clause",
    "synchronized_statement", "assert_statement", "yield_statement",
    "switch_block_statement_group", "switch_rule",
)
STMT_TYPES = STMT_TYPES_CPP + STMT_TYPES_JAVA

N_LEAF = len(LEAF_LABELS)
N_STMT_CPP = len(STMT_TYPES_CPP)
N_STMT = len(STMT_TYPES)
# Vị trí 0 trong `LABEL_NAMES` dùng cho code rỗng / không có node nào.
LABEL_NAMES = ("NONE",) + LEAF_LABELS + STMT_TYPES
VOCAB_SIZE = len(LABEL_NAMES)
LEAF_OFFSET = 1
STMT_OFFSET = 1 + N_LEAF

_STMT_INDEX = {t: i for i, t in enumerate(STMT_TYPES)}

_STR_TYPES = (
    "string_literal", "raw_string_literal", "concatenated_string",
    "string_content", "escape_sequence", "char_literal",
    "character_literal", "text_block",
)
_NUM_TYPES = (
    "number_literal", "integer_literal", "float_literal",
    "hex_integer_literal", "binary_integer_literal",
    "octal_integer_literal", "decimal_floating_literal",
    "hex_floating_literal", "char_literal", "char_literal_text",
    "decimal_integer_literal", "decimal_floating_point_literal",
    "hex_floating_point_literal",
)
_ID_TYPES = (
    "identifier", "field_identifier", "type_identifier",
    "primitive_type", "sized_type_specifier", "namespace_identifier",
    "statement_identifier",
    "integral_type", "floating_point_type", "boolean_type", "void_type",
)


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


def get_parser(language: str = "cpp") -> Parser:
    if language not in SUPPORTED:
        raise ValueError(f"chưa hỗ trợ ngôn ngữ {language!r}; có {sorted(SUPPORTED)}")
    if language == "cpp":
        try:
            import tree_sitter_cpp
            return Parser(Language(tree_sitter_cpp.language()))
        except Exception:
            import tree_sitter_language_pack as tslp
            return tslp.get_parser("cpp")
    if language == "java":
        try:
            import tree_sitter_java
            return Parser(Language(tree_sitter_java.language()))
        except Exception:
            import tree_sitter_language_pack as tslp
            return tslp.get_parser("java")
    import tree_sitter_python

    return Parser(Language(tree_sitter_python.language()))


def _leaf_kind(text: str, node_type: str) -> int:
    """Phân loại token theo nội dung khi tree-sitter không gắn nhãn sẵn.

    Trả -1 cho operator/punctuation/keyword → caller bỏ qua (giữ skeleton).
    """
    if node_type in _STR_TYPES:
        return LEAF_STR
    if node_type in _NUM_TYPES:
        return LEAF_NUM
    if node_type in _ID_TYPES:
        return LEAF_ID
    s = text.strip()
    if not s:
        return -1
    c = s[0]
    if c in "\"'":
        return LEAF_STR
    if c.isdigit():
        return LEAF_NUM
    if c.isalpha() or c == "_":
        return LEAF_ID
    return -1  # toán tử, ngoặc, dấu phẩy, keyword...


def extract_tree(
    code: str,
    parser: Parser | None = None,
    language: str = "cpp",
) -> dict[str, np.ndarray]:
    """Parse + abstraction thành cây statement dạng mảng phẳng.

    Cấu trúc trả về (mọi mảng cùng độ dài `n` = số node giữ lại):
      label   int16  — id nhãn trong `LABEL_NAMES`
      parent  int16  — chỉ số statement cha gần nhất; node gốc trỏ về chính nó
      is_leaf uint8  — 1 nếu node là lá
      n_leaf  int16  — số lá trong cây con (chỉ statement node có giá trị > 0)
      span_start int32 — byte offset bắt đầu node trong code gốc (tree-sitter)
      span_end   int32 — byte offset kết thúc (không bao gồm)
    Hai mảng span CHỈ phục vụ highlight evidence ở demo (map vector ngược ra
    code); pipeline train (compile/collate/store) đọc key tường minh nên thêm
    key không ảnh hưởng gì.
    """
    parser = parser or get_parser(language)
    root = parser.parse(code.encode("utf-8")).root_node

    labels: list[int] = []
    parents: list[int] = []
    is_leaf: list[int] = []
    n_leaf: list[int] = []
    span_start: list[int] = []
    span_end: list[int] = []

    def visit(node: Node, parent_idx: int) -> int:
        """Trả về số lá nằm trong node này; node giữ lại sẽ được append."""
        if node.type in _STMT_INDEX:
            idx = len(labels)
            labels.append(STMT_OFFSET + _STMT_INDEX[node.type])
            parents.append(parent_idx)
            is_leaf.append(0)
            n_leaf.append(0)
            span_start.append(node.start_byte)
            span_end.append(node.end_byte)
            got = 0
            for child in node.children:
                got += visit(child, idx)
            n_leaf[idx] = got
            return got

        # Không phải statement → flatten: bỏ node, giữ lá bên trong.
        got = 0
        for child in node.children:
            got += visit(child, parent_idx)
        if node.child_count == 0:
            kind = _leaf_kind(node.text.decode("utf-8", "replace"), node.type)
            if kind >= 0:
                idx = len(labels)
                labels.append(LEAF_OFFSET + kind)
                parents.append(parent_idx)
                is_leaf.append(1)
                n_leaf.append(1)
                span_start.append(node.start_byte)
                span_end.append(node.end_byte)
                got = 1
        return got

    if root.type in _STMT_INDEX:
        visit(root, 0)
    else:  # root hiếm (vd file chỉ có comment) → node ảo làm gốc
        labels, parents, is_leaf, n_leaf = [0], [0], [0], [0]
        span_start, span_end = [0], [0]
        for child in root.children:
            visit(child, 0)

    if not labels:  # code rỗng
        labels, parents, is_leaf, n_leaf = [0], [0], [0], [0]
        span_start, span_end = [0], [0]
    return {
        "label": np.asarray(labels, dtype=np.int16),
        "parent": np.asarray(parents, dtype=np.int64),
        "is_leaf": np.asarray(is_leaf, dtype=np.int64),
        "n_leaf": np.asarray(n_leaf, dtype=np.int16),
        "span_start": np.asarray(span_start, dtype=np.int64),
        "span_end": np.asarray(span_end, dtype=np.int64),
    }


def depth_of(parent: np.ndarray) -> np.ndarray:
    """Độ sâu từng node (root = 0).

    Do DFS pre-order nên cha luôn có chỉ số nhỏ hơn con → một vòng lặp là đủ.
    """
    p = np.asarray(parent, dtype=np.int64)
    depth = np.zeros(len(p), dtype=np.int64)
    for i in range(1, len(p)):
        depth[i] = depth[p[i]] + 1
    return depth


def tree_stats(tree: dict[str, np.ndarray]) -> dict:
    """Thống kê nhỏ để audit chất lượng AST (đưa vào report)."""
    depth = depth_of(tree["parent"])
    leaf = tree["is_leaf"] == 1
    return {
        "n_nodes": int(len(tree["label"])),
        "n_stmt": int((~leaf).sum()),
        "n_leaf": int(leaf.sum()),
        "max_depth": int(depth.max()),
        "mean_depth": round(float(depth.mean()), 2),
    }


# ---------------------------------------------------------------------------
# Compile: dựng sẵn mọi chỉ số RvNN cần, để forward chỉ còn vài phép tensor.
# Mọi bước O(n) / O(n·depth) đều làm MỘT LẦN ở đây, không lặp mỗi epoch.
#
# RvNN chỉ đệ quy trên **statement node** (không đệ quy trên lá). Lá chỉ đóng
# góp vào vector đầu vào của statement qua cặp (statement, lớp lá) — `anc_*`.
# Nhờ vậy batch chỉ cần S ≈ 55 statement/code thay vì 287 node/code.
# ---------------------------------------------------------------------------

_EMPTY_TREE = {
    "s_label": np.zeros(1, dtype=np.int64),
    "s_parent": np.zeros(1, dtype=np.int64),
    "s_depth": np.zeros(1, dtype=np.int64),
    "s_order": np.zeros(1, dtype=np.int64),
    "s_level_ptr": np.array([0, 1], dtype=np.int64),
    "anc_s": np.zeros(0, dtype=np.int64),
    "anc_c": np.zeros(0, dtype=np.int64),
    "top": np.zeros(1, dtype=np.int64),
    "n_stmt": np.int32(1),
    "n_top": np.int32(1),
}


def compile_tree(tree: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Dựng chỉ số RvNN trong statement space (không phải node space).

    Sinh ra (mọi mảng đã index trong không gian statement 0..S-1):
      s_label     (S,)    — loại statement, 0..N_STMT-1
      s_parent    (S,)    — statement cha; gốc trỏ về chính nó (0)
      s_depth     (S,)    — độ sâu trên cây statement
      s_order     (S,)    — chỉ số statement, depth **giảm dần**
      s_level_ptr (L+1,)  — biên tầng trong `s_order`; lát tầng j là
                            `s_order[ptr[j]:ptr[j+1]]` và ứng với depth (L-j)
      anc_s       (a,)    — statement (stmt-space) của từng cặp
      anc_c       (a,)    — lớp lá 0..2 của từng cặp
      top         (m,)    — statement cấp 1 (con của gốc), thứ tự DFS
      n_stmt / n_top     — số statement / số statement cấp 1
    """
    label = tree["label"].astype(np.int64)
    parent = tree["parent"].astype(np.int64)
    is_leaf = tree["is_leaf"].astype(np.int64)

    stmt_nodes = np.flatnonzero(is_leaf == 0)  # đã theo thứ tự DFS
    n_stmt = len(stmt_nodes)
    if n_stmt == 0:
        return {k: v.copy() for k, v in _EMPTY_TREE.items()}

    pos2stmt = np.full(len(label), -1, dtype=np.int64)
    pos2stmt[stmt_nodes] = np.arange(n_stmt, dtype=np.int64)
    # Node ảo (label 0 = NONE) chỉ có khi root không phải statement → label âm.
    s_label = np.maximum(label[stmt_nodes] - STMT_OFFSET, 0)

    def nearest_stmt(node: int) -> int:
        """Chỉ số statement cha gần nhất, hoặc -1 nếu parser trả cây lỗi.

        Lưu ý: node 0 **là** statement gốc (đã set `parent[0] = 0` làm sentinel),
        nên điều kiện dừng là `pos2stmt[p] < 0` chứ không phải `p == 0`. Nếu dùng
        `p == 0` thì mọi con trực tiếp của gốc sẽ bị coi là "không có cha" →
        tự tham chiếu → vòng lặp độ sâu treo vô hạn.
        """
        p = int(parent[node])
        hops = 0
        while pos2stmt[p] < 0:
            p = int(parent[p])
            hops += 1
            if hops > len(label):  # chống treo nếu parser trả cây lỗi
                return -1
        return int(pos2stmt[p])

    # --- cây statement ----------------------------------------------------
    s_parent = np.empty(n_stmt, dtype=np.int64)
    for i, nd in enumerate(stmt_nodes):
        p = nearest_stmt(int(nd))
        s_parent[i] = i if p < 0 else p

    # Độ sâu bằng MỘT pass tiến: cha luôn DFS trước nên `s_parent[i] < i` với
    # i>0. Không đi ngược từng node (gốc tự tham chiếu nên vòng "lên tới self"
    # không bao giờ dừng với node con của gốc).
    if n_stmt > 1 and not np.all(s_parent[1:] < np.arange(1, n_stmt)):
        raise RuntimeError("compile_tree: cây statement không phải DFS pre-order")
    s_depth = np.zeros(n_stmt, dtype=np.int64)
    for i in range(1, n_stmt):
        s_depth[i] = s_depth[s_parent[i]] + 1

    # --- cặp (statement, lớp lá) -----------------------------------------
    a_s, a_c = [], []
    for nd in np.flatnonzero(is_leaf == 1):
        cls = int(label[nd]) - LEAF_OFFSET
        tgt = nearest_stmt(int(nd))
        a_s.append(0 if tgt < 0 else tgt)  # lá ngoài statement → gán cho gốc
        a_c.append(cls)
    anc_s = np.asarray(a_s, dtype=np.int64)
    anc_c = np.asarray(a_c, dtype=np.int64)

    # --- thứ tự duyệt tầng ------------------------------------------------
    s_order = np.argsort(-s_depth, kind="stable").astype(np.int64)
    n_levels = int(s_depth.max()) + 1
    s_level_ptr = np.zeros(n_levels + 1, dtype=np.int64)
    s_level_ptr[1:] = np.cumsum(np.bincount(s_depth, minlength=n_levels)[::-1])

    # Statement cấp 1 = con của gốc. Gốc có `s_parent[0] == 0` (tự tham chiếu
    # làm sentinel) nên phải loại chính nó — nếu không, `top` sẽ chứa cả gốc và
    # bi-GRU sẽ ăn vào vector tổng của toàn bộ file.
    top = np.flatnonzero(s_parent == 0).astype(np.int64)
    top = top[top != 0]
    if len(top) == 0:
        top = np.zeros(1, dtype=np.int64)

    return {
        "s_label": s_label,
        "s_parent": s_parent,
        "s_depth": s_depth,
        "s_order": s_order,
        "s_level_ptr": s_level_ptr,
        "anc_s": anc_s,
        "anc_c": anc_c,
        "top": top,
        "n_stmt": np.int32(n_stmt),
        "n_top": np.int32(len(top)),
    }


def collate_trees(compiled: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Ghép nhiều cây thành một batch phẳng trong statement space.

    Batch gồm `2 * n_pairs` cây theo thứ tự [a0, b0, a1, b1, ...].
    `s_order` được sắp lại **toàn cục** theo depth giảm dần để RvNN duyệt tầng
    bằng lát liên tục (không nối mải các order cục bộ).
    """
    if not compiled:
        raise ValueError("collate_trees: batch rỗng")
    lens = [int(c["n_stmt"]) for c in compiled]
    offs = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    s_label = np.concatenate([c["s_label"] for c in compiled])
    s_parent = np.concatenate(
        [c["s_parent"] + offs[i] for i, c in enumerate(compiled)])
    # `s_depth` là GIÁ TRỊ, không phải chỉ số → không offset. Offset nó sẽ phá
    # vỡ thứ tự tầng khi sort toàn cục.
    s_depth = np.concatenate([c["s_depth"] for c in compiled])
    anc_s = np.concatenate([c["anc_s"] + offs[i] for i, c in enumerate(compiled)])
    anc_c = np.concatenate([c["anc_c"] for c in compiled])

    s_order = np.argsort(-s_depth, kind="stable").astype(np.int64)
    n_levels = int(s_depth.max()) + 1
    s_level_ptr = np.zeros(n_levels + 1, dtype=np.int64)
    s_level_ptr[1:] = np.cumsum(np.bincount(s_depth, minlength=n_levels)[::-1])

    n_top_max = max(int(c["n_top"]) for c in compiled) or 1
    b = len(compiled)
    top_idx = np.zeros((b, n_top_max), dtype=np.int64)
    top_mask = np.zeros((b, n_top_max), dtype=np.uint8)
    for i, c in enumerate(compiled):
        m = int(c["n_top"])
        if m:
            top_idx[i, :m] = c["top"] + offs[i]
            top_mask[i, :m] = 1

    return {
        "s_label": s_label,
        "s_parent": s_parent,
        "s_depth": s_depth,
        "s_order": s_order,
        "s_level_ptr": s_level_ptr,
        "anc_s": anc_s,
        "anc_c": anc_c,
        "top_idx": top_idx,
        "top_mask": top_mask,
        "stmt_offs": offs,
        "batch_size": b,
    }
