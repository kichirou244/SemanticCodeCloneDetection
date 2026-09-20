<a id="top"></a>

# Tổng quan project (project_overview.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Mục tiêu](#1-muc-tieu)
- [2. Phạm vi](#2-pham-vi)
- [3. Ma trận model core](#3-ma-tran-model-core)
- [4. Pipeline dữ liệu](#4-pipeline-du-lieu)
- [5. Nguyên tắc nghiên cứu](#5-nguyen-tac-nghien-cuu)
- [6. Lịch sử thay đổi](#6-lich-su-thay-doi)

---

## 1. Mục tiêu

Xây dựng và đánh giá các phương pháp AI/Deep Learning phát hiện hai đoạn code
cùng/tương đương ngữ nghĩa-chức năng, kể cả khi khác tên biến, tên hàm, AST,
control flow hoặc cách triển khai. Output phục vụ trực tiếp bài báo khoa học.

[🔝 Quay lại mục lục](#top)

---

## 2. Phạm vi

- Ngôn ngữ: **C/C++** (POJ-104) và **Python** (CodeNet Python). Không mở rộng
  sang Java/C#/JavaScript khi chưa có yêu cầu rõ ràng.
- Core experiment: **1 baseline (B1) + 5 model (M1–M5)**, xem bảng dưới.
- Phase INIT: chưa train đầy đủ; chỉ loaders, audit, leakage, interface,
  smoke test baseline/transformer, feasibility AST/CPG.

[🔝 Quay lại mục lục](#top)

---

## 3. Ma trận model core

| ID | Phương pháp | Nhóm | Trạng thái INIT |
| :--- | :--- | :--- | :--- |
| B1 | TF-IDF + Cosine | Baseline | Chạy được (smoke) |
| M1 | CodeBERT | Transformer | Smoke embedding |
| M2 | GraphCodeBERT | Transformer | Smoke embedding |
| M3 | ASTNN | AST | Stub + feasibility |
| M4 | CPG-GNN (`gat`) | Program Graph | Stub + feasibility |
| M5 | GraphCodeBERT + CPG-GNN (`concat_mlp`) | Hybrid | Stub |

[🔝 Quay lại mục lục](#top)

---

## 4. Pipeline dữ liệu

```text
Raw → Audit → Cleaning → Deduplication → Normalization → Leakage Check
→ Split (theo problem) → Pair Generation → Train → Validation → Test
```

Chi tiết: `docs/dataset_protocol.md`, `docs/leakage_prevention.md`.

[🔝 Quay lại mục lục](#top)

---

## 5. Nguyên tắc nghiên cứu

1. Reproducibility: seed cố định, config YAML, log môi trường.
2. Fairness: cùng split/protocol/metric cho mọi model.
3. Không leakage: split theo problem trước augmentation/transformation.
4. Code + Docs + Reports phát triển song song, paper-ready.

[🔝 Quay lại mục lục](#top)

---

## 6. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
