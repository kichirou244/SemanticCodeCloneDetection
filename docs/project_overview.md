<a id="top"></a>

# Tổng quan project (project_overview.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.2.0 |
| **Ngày cập nhật** | 2026-09-28 |
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

- Ngôn ngữ: **C/C++ (POJ-104) — phạm vi duy nhất** (DEC-019). Khối Python
  (CodeNet) đã hủy vì quota compute; dataset vẫn giữ trong repo để mở lại được.
  Không mở rộng sang Java/C#/JavaScript.
- Core experiment: **1 baseline (B1) + 5 model (M1–M5)** trên POJ-104, xem bảng
  dưới (EXP-001…EXP-006).
- Trạng thái hiện tại: EXP-001, EXP-002 COMPLETED; M1/M2/M3 đã implement + test
  (EXP-003, EXP-004 chờ train); M4/M5 còn là stub.

[🔝 Quay lại mục lục](#top)

---

## 3. Ma trận model core

| ID | Phương pháp | Nhóm | Trạng thái (2026-09-28) |
| :--- | :--- | :--- | :--- |
| B1 | TF-IDF + Cosine | Baseline | **COMPLETED** (F1 0.673) |
| M1 | CodeBERT | Transformer | **COMPLETED** (F1 0.910, EXP-002) |
| M2 | GraphCodeBERT | Transformer | Xong, chờ train (EXP-003), dùng chung code path M1 |
| M3 | ASTNN | AST | Xong, chờ train (EXP-004), ~0,6h |
| M4 | CPG-GNN (`gat`) | Program Graph | Stub + feasibility (cần Joern, DEC-009) |
| M5 | GraphCodeBERT + CPG-GNN (`concat_mlp`) | Hybrid | Stub (cần M2 + M4) |

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
| 2 | 1.1.0 | 2026-09-27 | Cập nhật ma trận model: M1/M2/M3 đã implement + test; M4/M5 còn stub (DEC-019/020/021). |
| 3 | 1.2.0 | 2026-09-28 | M1 (EXP-002) → **COMPLETED**, test F1 0.910 (ROC-AUC 0.973), vượt B1 +0.237 F1 (reports/011, DEC-022). |
