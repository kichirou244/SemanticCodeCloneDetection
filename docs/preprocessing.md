<a id="top"></a>

# Preprocessing & Representation & Model protocol (preprocessing.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.1.0 |
| **Ngày cập nhật** | 2026-09-27 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Preprocessing](#1-preprocessing)
- [2. Representation](#2-representation)
- [3. Model](#3-model)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Preprocessing

Fit (TF-IDF, scaler nếu có) chỉ trên train; transform val/test
(`src/models/baseline/tfidf.py::fit`). Mọi bước ghi seed + version dataset.

[🔝 Quay lại mục lục](#top)

---

## 2. Representation

- Token: tokenizer của từng Transformer, `max_length` trong config. M1/M2 dùng
  `max_length: 256` ⇒ **60,5% token bị truncate** trên POJ-104.
- AST: tree-sitter, abstraction theo language (`src/representations/ast/`).
  Đã implement cho C++ (`extract.py`): parse → `ID`/`NUM`/`STR` → **rừng cây
  statement**, mỗi lá gắn với statement cha gần nhất. Cây được cache dạng
  flat arrays trong `.npz` (`store.py`), compile một lần cho cả 3 epoch.
- CPG: AST + CFG + Data Flow, chuẩn hóa graph; C++ và Python dùng backend
  riêng qua cùng abstraction (`src/representations/cpg/`). Không hard-code
  logic một language vào pipeline core. **Chưa implement** — cần Joern cho C++
  (DEC-009).

## 3. Model

| M | Model | Trạng thái | File |
| :--- | :--- | :--- | :--- |
| M1 | CodeBERT | xong | `src/models/transformer_pair.py` (generic) + wrapper `codebert_pair.py` |
| M2 | GraphCodeBERT | xong | Cùng class, chỉ khác `hf_id` (DEC-020) |
| M3 | ASTNN | xong | `src/models/astnn/model.py` (RvNN vectorize + GRU) |
| M4 | CPG-GNN | stub | Backbone chính `gat` |
| M5 | Hybrid | stub | Fusion `concat_mlp` |

- M1/M2 dùng chung `TransformerPairClassifier`: đầu vào `hA ⊕ hB ⊕ |hA−hB|`
  → MLP → 1 logit, huấn luyện `BCEWithLogitsLoss`, threshold chọn trên
  validation F1 (DEC-014/016).
- M3 dùng `ASTNNPairClassifier`: mỗi side là (GRU trên statement) ⊕ (biểu diễn
  RvNN trên cây statement). **Không truncate** — giữ nguyên cây, nên không chịu
  giới hạn 256 token mà M1/M2 đang chịu.
- Đổi backbone/fusion của M4/M5 phải ghi decision log.

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo gộp preprocessing/representation/model ở INIT. |
| 2 | 1.1.0 | 2026-09-27 | Cập nhật §2–3: M1/M2 đã có classifier thật, M3 ASTNN đã implement (rừng statement + RvNN vectorize + cache), CPG còn stub. |
