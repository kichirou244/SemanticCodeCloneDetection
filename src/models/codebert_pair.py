"""Alias giữ tương thích — M1 CodeBERT.

Implementation thật đã chuyển sang `src/models/transformer_pair.py` để M2
GraphCodeBERT dùng **chung code** (DEC-016/RQ1 cần implementation gap = 0).

Module này giữ nguyên tên cũ để EXP-002 còn tái lập được và test cũ không đổi.
"""

from src.models.transformer_pair import (
    TransformerPairClassifier,
    collate_pair_batch,
    load_tokenizer,
    tokenize_side,
)

# Tên lớp cũ = tên lớp mới (alias, không phải subclass).
CodeBERTPairClassifier = TransformerPairClassifier

__all__ = [
    "CodeBERTPairClassifier",
    "TransformerPairClassifier",
    "collate_pair_batch",
    "load_tokenizer",
    "tokenize_side",
]
