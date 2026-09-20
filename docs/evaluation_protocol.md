<a id="top"></a>

# Evaluation protocol (evaluation_protocol.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Classification](#1-classification)
- [2. Retrieval](#2-retrieval)
- [3. Error analysis](#3-error-analysis)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Classification

Task: cặp (A, B) → Clone/Non-clone. Metric chính: **Precision, Recall, F1**
(threshold 0.5 mặc định, `configs/models/*.yaml`); phụ: ROC-AUC, PR-AUC.
Implement: `src/evaluation/metrics.py::classification_metrics`.

[🔝 Quay lại mục lục](#top)

---

## 2. Retrieval

Task: query → rank candidates. Metrics: MAP@R, MRR, Recall@K, Precision@K
(`mrr`, `recall_at_k` đã có; MAP@R bổ sung ở phase sau nếu dùng retrieval).

[🔝 Quay lại mục lục](#top)

---

## 3. Error analysis

Lưu FP/FN kèm `code_a, code_b, ground_truth, prediction, confidence, model,
language, dataset, experiment_id` qua `src/analysis/errors.py` vào
`results/errors/`. Phân tích nhóm lỗi: lexical cao/semantics khác, lexical
thấp/semantics giống, AST giống/chức năng khác, graph khác/chức năng giống,
lỗi Transformer/Graph/Hybrid.

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
