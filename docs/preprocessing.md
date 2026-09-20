<a id="top"></a>

# Preprocessing & Representation & Model protocol (preprocessing.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
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

- Token: tokenizer của từng Transformer, `max_length` trong config.
- AST: tree-sitter, abstraction theo language (`src/representations/ast/`).
- CPG: AST + CFG + Data Flow, chuẩn hóa graph; C++ và Python dùng backend
  riêng qua cùng abstraction (`src/representations/cpg/`). Không hard-code
  logic một language vào pipeline core.

[🔝 Quay lại mục lục](#top)

---

## 3. Model

- M1/M2: `src/models/transformer/embedder.py` (INIT: embedding; classifier ở
  phase sau).
- M3/M4/M5: stub đã khóa interface; M4 backbone chính `gat`, M5 fusion
  `concat_mlp`. Đổi backbone/fusion phải ghi decision log.

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo gộp preprocessing/representation/model ở INIT. |
