<a id="top"></a>

# Transformation & Ablation protocol (transformation_protocol.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Transformation (RQ4)](#1-transformation-rq4)
- [2. Ablation](#2-ablation)
- [3. Lịch sử thay đổi](#3-lich-su-thay-doi)

---

## 1. Transformation (RQ4)

8 loại đã khóa tên trong `src/transformations/registry.py`. Engine implement ở
phase sau. Quy tắc leakage: transform **sau** khi split; không để bản gốc ở
train và bản transform ở test khi đo generalization.

[🔝 Quay lại mục lục](#top)

---

## 2. Ablation

Phase sau: ablation data-flow của M2, từng thành phần CPG (AST/CFG/DF) của M4,
và fusion của M5. Mỗi ablation là một experiment riêng trong registry.

[🔝 Quay lại mục lục](#top)

---

## 3. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo gộp transformation/ablation ở INIT. |
