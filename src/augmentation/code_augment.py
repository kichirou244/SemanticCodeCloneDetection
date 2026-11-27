"""Biến đổi code giữ nguyên ngữ nghĩa/phán đoán clone.

Chiến lược an toàn (không đổi cấu trúc điều khiển, không xóa câu lệnh):
- Xóa comment (`//`, `/* */`) — comment không ảnh hưởng ngữ nghĩa.
- Đổi tên identifier nhất quán *trong từng hàm* (kể cả tên method/type) —
  với cặp clone, hai hàm được rename độc lập nên model không thể bám vào
  tên biến trùng nhau; đây chính là tín hiệu giúp giảm FN kiểu "cùng ngữ nghĩa,
  khác bề mặt".
- Chuẩn hóa whitespace.

Không dùng hoán vị câu lệnh vì có thể đổi ngữ nghĩa (Side effects).
"""

from __future__ import annotations

import random
import re
from typing import Any

_IDENT_RE = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b")

_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
    "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
    "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
    "volatile", "while", "true", "false", "null",
}


def strip_comments(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    code = re.sub(r"//[^\n]*", " ", code)
    return code


def rename_identifiers(code: str, rng: random.Random) -> str:
    """Đổi mỗi identifier thành `v0`, `v1`, ... theo thứ tự xuất hiện."""
    mapping: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _KEYWORDS:
            return name
        if name not in mapping:
            mapping[name] = f"v{len(mapping)}"
        return mapping[name]

    return _IDENT_RE.sub(repl, code)


def normalize_whitespace(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.splitlines()).strip()


def augment_pair(
    func1: str,
    func2: str,
    *,
    rng: random.Random,
    rename_prob: float = 0.5,
    strip_comments_prob: float = 0.5,
) -> tuple[str, str]:
    """Augment độc lập 2 vế của một cặp (vẫn giữ nhãn clone/non-clone)."""
    out1, out2 = func1, func2
    if strip_comments_prob > 0 and rng.random() < strip_comments_prob:
        out1, out2 = strip_comments(out1), strip_comments(out2)
    if rename_prob > 0 and rng.random() < rename_prob:
        out1 = rename_identifiers(out1, rng)
        out2 = rename_identifiers(out2, rng)
    return normalize_whitespace(out1), normalize_whitespace(out2)


def make_augmented_dataset(
    dataset: Any,
    *,
    rename_prob: float = 0.5,
    strip_comments_prob: float = 0.5,
    seed: int = 42,
) -> Any:
    """Wrapper Dataset: augment từng mẫu khi `__getitem__` (chỉ dùng cho train)."""
    from dataclasses import replace

    from torch.utils.data import Dataset

    class AugmentedPairDataset(Dataset):
        def __init__(self, base: Any) -> None:
            self.base = base
            self.rng = random.Random(seed)

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, idx: int) -> Any:
            sample = self.base[idx]
            f1, f2 = augment_pair(
                sample.func1,
                sample.func2,
                rng=self.rng,
                rename_prob=rename_prob,
                strip_comments_prob=strip_comments_prob,
            )
            return replace(sample, func1=f1, func2=f2)

    return AugmentedPairDataset(dataset)
