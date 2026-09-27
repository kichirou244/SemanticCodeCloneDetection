<a id="top"></a>

# Câu hỏi nghiên cứu (research_questions.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.1.0 |
| **Ngày cập nhật** | 2026-09-27 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. RQ1 — Representation](#1-rq1--representation)
- [2. RQ2 — Transformer vs Program Graph](#2-rq2--transformer-vs-program-graph)
- [3. RQ3 — Hybrid Representation](#3-rq3--hybrid-representation)
- [4. RQ4 — Robustness](#4-rq4--robustness)
- [5. Phạm vi ngôn ngữ](#5-phạm-vi-ngôn-ngữ)
- [6. Lịch sử thay đổi](#6-lich-su-thay-doi)

---

## 1. RQ1 — Representation

> How do token-based, AST-based, and program-graph-based representations affect
> semantic code clone detection?

So sánh: CodeBERT (M1) vs GraphCodeBERT (M2) vs ASTNN (M3) vs CPG-GNN (M4).

Trả lời trên POJ-104 (C/C++). Không phụ thuộc ngôn ngữ — xem DEC-019.

[🔝 Quay lại mục lục](#top)

---

## 2. RQ2 — Transformer vs Program Graph

> Does explicit program-graph information improve semantic clone detection
> compared with pretrained code Transformers?

So sánh: GraphCodeBERT (M2) vs CPG-GNN (M4) trên cùng split và metric.

[🔝 Quay lại mục lục](#top)

---

## 3. RQ3 — Hybrid Representation

> Does combining GraphCodeBERT and CPG-GNN provide complementary information
> for clone detection?

So sánh: M2 vs M4 vs M5 (fusion `concat_mlp` ở INIT).

[🔝 Quay lại mục lục](#top)

---

## 4. RQ4 — Robustness

> How robust are the methods under semantics-preserving source-code
> transformations?

8 transformation đã khóa tên trong `src/transformations/registry.py`
(identifier renaming, function renaming, formatting, dead-code insertion,
statement restructuring, control-flow tương đương, loop transformation,
variable substitution). Engine cài đặt ở phase sau; split/augment phải tuân
`docs/leakage_prevention.md`.

[🔝 Quay lại mục lục](#top)

---

## 5. Phạm vi ngôn ngữ

Cả bốn RQ được trả lời trên **POJ-104 (C/C++)** — chúng so sánh *cách biểu diễn
code*, không phụ thuộc ngôn ngữ cụ thể (DEC-019).

Không thí nghiệm trên Python (CodeNet) trong phạm vi hiện tại. Hệ quả cần nêu
rõ khi viết bài: **chưa có bằng chứng generalization sang ngôn ngữ khác**. Nêu ở
mục limitation thay vì bỏ trống. Dataset CodeNet đã audit + build sẵn
(`data/manifests/codenet_python.json`) nên có thể mở rộng lại sau nếu reviewer
yêu cầu.

[🔝 Quay lại mục lục](#top)

---

## 6. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
| 2 | 1.1.0 | 2026-09-27 | DEC-019: thêm mục phạm vi ngôn ngữ (chỉ C/C++, nêu rõ thiếu cross-language evidence). |
