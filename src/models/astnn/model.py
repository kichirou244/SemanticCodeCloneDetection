"""M3 — ASTNN (Zhang et al. 2019), bản vectorize hoàn toàn.

Kiến trúc (giữ đúng bài báo):
  1. **Abstraction** (xem `src/representations/ast/extract.py`): AST rút gọn thành
     cây statement, lá chỉ còn ID/NUM/STR.
  2. **RvNN** — mã hóa cây statement từ dưới lên:
         h(v) = tanh( W_s·s(v) + W_x·x(v) + W_h·Σ_children h(c) )
     trong đó `x(v)` là vector trung bình các lá thuộc cây con của v.
  3. **bi-GRU** trên chuỗi các vector statement cấp 1 → vector đặc trưng hàm.
  4. Pair head giống hệt M1/M2: `[hA ; hB ; |hA−hB|] → MLP → logit`.

Vì sao vectorize (điểm mấu chốt cho thời gian chạy):
  Bản gốc của ASTNN lặp Python từng node để đệ quy. Với POJ-104 (tb ~55
  statement, ~232 lá mỗi code) và 332k pairs × 3 epoch thì phần đệ quy đó chiếm
  **giờ CPU**, không phải GPU. Ở đây toàn bộ cây của cả batch được gộp thành
  các mảng phẳng và duyệt **theo tầng độ sâu**: mỗi tầng là 1 matmul + 1
  `index_add_`, tổng ~n_levels ≈ 6-20 phép tensor cho cả batch thay vì
  hàng trăm nghìn lệnh Python. Đây là bước quyết định 100-1000×.

Mọi chỉ số cần cho RvNN được `extract.compile_tree` + `extract.collate_trees`
tính **một lần** trước khi train, nên forward pass không còn việc duyệt cây.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.representations.ast.extract import N_LEAF, N_STMT


def to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Chuyển batch numpy của `collate_trees` sang tensor trên `device`."""
    keys = ("s_label", "s_parent", "s_order", "anc_s", "anc_c", "top_idx")
    out = {k: torch.as_tensor(batch[k], dtype=torch.long, device=device)
           for k in keys}
    out["s_level_ptr"] = torch.as_tensor(batch["s_level_ptr"], dtype=torch.long,
                                         device=device)
    out["top_mask"] = torch.as_tensor(batch["top_mask"], dtype=torch.float32,
                                      device=device)
    out["n_levels"] = int(len(batch["s_level_ptr"]) - 1)
    out["batch_size"] = int(batch["batch_size"])
    return out


class VectorizedRvNN(nn.Module):
    """RvNN duyệt tầng theo độ sâu, batch hoàn toàn trên GPU."""

    def __init__(self, hidden: int = 200):
        super().__init__()
        self.hidden = hidden
        self.stmt_emb = nn.Embedding(N_STMT, hidden)
        self.leaf_emb = nn.Embedding(N_LEAF, hidden)
        self.W_s = nn.Linear(hidden, hidden)
        self.W_x = nn.Linear(hidden, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden)
        self.act = nn.Tanh()

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        s_label = b["s_label"]
        s_parent = b["s_parent"]
        order = b["s_order"]
        anc_s = b["anc_s"]
        anc_c = b["anc_c"]
        level_ptr = b["s_level_ptr"]
        n_stmt = s_label.shape[0]

        # --- x(v): tổng vector lá theo cây con, rồi chiếu W_x -------------
        # index_add_ gom theo (statement, lớp lá) đã dựng sẵn -> 1 phép.
        if anc_s.numel() > 0:
            leaf_h = self.leaf_emb(anc_c)                       # (a, H)
            x = torch.zeros(n_stmt, self.hidden, dtype=leaf_h.dtype,
                            device=leaf_h.device)
            x = x.index_add(0, anc_s, self.W_x(leaf_h))
        else:
            x = torch.zeros(n_stmt, self.hidden, dtype=self.W_s.weight.dtype,
                            device=s_label.device)

        # --- s(v): embedding loại statement --------------------------------
        x = x + self.W_s(self.stmt_emb(s_label))

        # --- đệ quy từ lá lên gốc, theo tầng ------------------------------
        # Hai tensor phải tách biệt:
        #   h_acc[v] = tổng h(c) của các con  -> ĐẦU VÀO của W_h ở tầng sau
        #   h_out[v] = h(v) sau tanh          -> KẾT QUẢ của node v
        # Gộp chúng là một (index_add đè luôn lên chính h_acc) khiến hàm trả về
        # tổng con thay vì h(v) và mọi node lá ra 0 — vẫn chạy, vẫn số hữu ích,
        # nên test finiteness bình thường KHÔNG bắt được. Chỉ test so với bản
        # đệ quy tham chiếu mới lộ ra.
        #
        # Hàng `n_stmt` là "thùng rác": node gốc tự tham chiếu nên đổ vào đó
        # thay vì cộng vào chính nó (giữ h_acc của gốc là sạch).
        h_acc = torch.zeros(n_stmt + 1, self.hidden, dtype=x.dtype, device=x.device)
        h_out = torch.zeros(n_stmt, self.hidden, dtype=x.dtype, device=x.device)
        for j in range(b["n_levels"]):
            lo = int(level_ptr[j])
            hi = int(level_ptr[j + 1])
            if hi <= lo:
                continue
            nodes = order[lo:hi]
            h = self.act(x.index_select(0, nodes)
                         + self.W_h(h_acc.index_select(0, nodes)))
            h_out = h_out.index_copy(0, nodes, h)
            par = s_parent.index_select(0, nodes)
            par = torch.where(par == nodes, torch.full_like(par, n_stmt), par)
            h_acc = h_acc.index_add(0, par, h)
        return h_out


class ASTNNPairClassifier(nn.Module):
    """M3 ASTNN cho bài toán clone *pair* — cùng head với M1/M2 (DEC-016)."""

    def __init__(self, hidden: int = 200, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.rvnn = VectorizedRvNN(hidden)
        # Bi-GRU 1 lớp, 2 hướng, mỗi hướng hidden//2 → encode_dim = hidden.
        self.enc_dim = 2 * (hidden // 2)
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True,
                          bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Head giống hệt M1/M2: [hA ; hB ; |hA−hB|] → 3 * enc_dim
        self.head = nn.Sequential(
            nn.Linear(3 * self.enc_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_side(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        h_stmt = self.rvnn(b)                                  # (S, H)
        seq = h_stmt.index_select(0, b["top_idx"].reshape(-1)).view(
            b["batch_size"], b["top_idx"].shape[1], self.hidden)
        lengths = b["top_mask"].sum(dim=1).long().clamp(min=1).cpu()
        packed = pack_padded_sequence(
            seq, lengths, batch_first=True, enforce_sorted=False)
        # Với input PackedSequence, nn.GRU trả 2-tuple (PackedSequence, h_n)
        # — KHÁC LSTM là (output, (h_n, h_c)). Unpack 3 tầng sẽ cắt mất chiều
        # batch của h_n và âm thầm ra shape sai.
        _, h_n = self.gru(packed)
        # h_n: (num_directions * num_layers, batch, hidden//2); bi-GRU 1 lớp →
        # h_n[0] = trạng thái cuối hướng forward, h_n[1] = hướng backward.
        return torch.cat([h_n[0], h_n[1]], dim=-1)            # (B, enc_dim)

    def forward(
        self,
        batch_a: dict[str, torch.Tensor],
        batch_b: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        h_a = self.encode_side(batch_a)
        h_b = self.encode_side(batch_b)
        pair = torch.cat([h_a, h_b, torch.abs(h_a - h_b)], dim=-1)
        return self.head(self.dropout(pair)).squeeze(-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
