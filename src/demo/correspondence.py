"""Bắt cặp tương ứng giữa hai phía bằng embedding của chính model.

Mỗi model similarity chỉ cho một điểm số, không nói đoạn nào khớp đoạn nào.
Module này cho mỗi model tự bắt cặp: đơn vị phía A tìm bạn giống nhất phía B
và ngược lại, chỉ giữ cặp hai chiều đều chọn nhau, sắp theo cosine giảm dần.

Quy ước trả về: list `(i, j, cosine)` theo chỉ số hàng của hai ma trận vector
đã chuẩn hoá. Bên gọi tự đổi chỉ số hàng thành span ký tự của đơn vị đó.
"""

from __future__ import annotations

import torch


def unit_rows(h: torch.Tensor) -> torch.Tensor:
    """Chuẩn hoá từng hàng về vector đơn vị, hàng 0 giữ 0."""
    return h.float() / h.float().norm(dim=1, keepdim=True).clamp(min=1e-9)


def word_vectors(
    hidden: torch.Tensor,
    word_ids: list[int | None],
    offsets: list[tuple[int, int]],
) -> tuple[torch.Tensor, list[list[int]]]:
    """Gom vector subword thành vector từ bằng trung bình.

    Bỏ token đặc biệt (`word_id` None) và span rỗng. Trả về
    (ma trận đã chuẩn hoá theo hàng, spans ký tự từng từ).
    """
    by_word: dict[int, list[int]] = {}
    for i, w in enumerate(word_ids):
        s, e = offsets[i]
        if w is None or e <= s:
            continue
        by_word.setdefault(w, []).append(i)
    vecs, spans = [], []
    for w in sorted(by_word):
        idx = by_word[w]
        vecs.append(hidden[idx].mean(dim=0))
        s = min(offsets[i][0] for i in idx)
        e = max(offsets[i][1] for i in idx)
        spans.append([s, e])
    if not vecs:
        return torch.zeros((0, hidden.shape[-1])), []
    return unit_rows(torch.stack(vecs)), spans


def mutual_best(
    na: torch.Tensor,
    nb: torch.Tensor,
    cap: int = 30,
) -> list[tuple[int, int, float]]:
    """Cặp hai chiều đều chọn nhau trên cosine, sắp theo cosine giảm dần.

    `na`, `nb` đã chuẩn hoá theo hàng. Chỉ giữ cặp cosine > 0. Hoà trong
    argmax thì `argmax` lấy chỉ số nhỏ nhất: tất định. Cắt ở `cap` cặp đầu.
    """
    if na.shape[0] == 0 or nb.shape[0] == 0:
        return []
    cos = na @ nb.T
    best_b = cos.argmax(dim=1)
    best_a = cos.argmax(dim=0)
    out: list[tuple[int, int, float]] = []
    for i in range(cos.shape[0]):
        j = int(best_b[i])
        if int(best_a[j]) == i and float(cos[i, j]) > 0:
            out.append((i, j, float(cos[i, j])))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out[:cap]
