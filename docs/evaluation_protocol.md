<a id="top"></a>

# Evaluation protocol (evaluation_protocol.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.1.0 |
| **Ngày cập nhật** | 2026-10-17 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Classification](#1-classification)
- [2. Retrieval](#2-retrieval)
- [3. Error analysis](#3-error-analysis)
- [4. Chọn phương pháp cuối (EXP-FINAL-001)](#4-chon-phuong-phap-cuoi-exp-final-001)
- [5. Lịch sử thay đổi](#5-lich-su-thay-doi)

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

## 4. Chọn phương pháp cuối (EXP-FINAL-001)

Mục đích: kết thúc giai đoạn huấn luyện bằng một luật chọn rõ ràng, chạy lại được,
rồi đóng băng thành pipeline cuối (eval → chọn → inference).

- Config: `configs/final_pipeline.yaml` (khai báo candidate runs + luật chọn + model
  cho bước suy luận). Code: `src/evaluation/leaderboard.py`,
  `scripts/final_pipeline.py`. Report: `reports/014_final_selection.md`.
- **Luật chọn: `validation.f1`** (không dùng test để chọn — test chỉ để báo cáo,
  DEC-022). Chỉ run `role: candidate` mới được chọn; baseline chỉ để đối chiếu.
- **Ngưỡng đóng băng** theo từng run từ `threshold.json`; không dò lại ngưỡng trên
  test (dò lại là rò rỉ nhãn và làm CI hẹp giả tạo).
- **Lượng hoá độ bất định:** winner vs á quân so trên **chính tập đã dùng để chọn**
  bằng McNemar (exact binomial) và paired bootstrap CI 95% của ΔF1 — tái dùng
  `src/evaluation/uncertainty.py`, không cài đặt lại.
- **Pipeline chỉ cảnh báo, không tự loại:** ngưỡng quá mỏng, chênh lệch quá sát,
  val/test xếp hạng khác nhau, hai kiểm định bất đồng (`yes`/`no`/`mixed`) đều được
  ghi vào `selection.json`. Quyết định cuối là của con người.

```bash
python scripts/final_pipeline.py select                      # xếp hạng + chọn (CPU, không cần GPU)
python scripts/final_pipeline.py detect --a a.cpp --b b.cpp  # suy luận bằng model đã chọn
```

`select` ghi `results/final_pipeline/{leaderboard.json,leaderboard.md,selection.json}`.

[🔝 Quay lại mục lục](#top)

---

## 5. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
| 2 | 1.1.0 | 2026-10-17 | Thêm §4 EXP-FINAL-001: luật chọn phương pháp cuối theo `validation.f1`, ngưỡng đóng băng, McNemar + paired bootstrap tái dùng `uncertainty.py`; pipeline `select`/`detect` (`scripts/final_pipeline.py`). |
