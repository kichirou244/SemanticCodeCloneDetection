<a id="top"></a>

# Dataset protocol (dataset_protocol.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Dataset core](#1-dataset-core)
- [2. Audit bắt buộc](#2-audit-bat-buoc)
- [3. Cleaning & dedup](#3-cleaning--dedup)
- [4. Split](#4-split)
- [5. Pair generation](#5-pair-generation)
- [6. Lịch sử thay đổi](#6-lich-su-thay-doi)

---

## 1. Dataset core

- C/C++: POJ-104 duy nhất (`google/code_x_glue_cc_clone_detection_poj104`).
  Không OJClone/BigCloneBench ở core.
- **Schema POJ-104 đã chốt (audit full 2026-09-20): keys `[id, code, label]`** —
  field code là `code`, problem là `label`. Split chính thức
  train 32500 / validation 8500 / test 12000 nhưng **bị bác (DEC-007)**.
- Python: CodeNet Python duy nhất (`windchimeran/codenet_python`).
- **Schema CodeNet đã chốt (audit 20k streaming 2026-09-20): keys
  `[submission_id, problem_id, language, status, code_size, cpu_time, memory,
  code, input, output, problem_description]`** — có `status` để lọc
  accepted-only (quyết ở phase pair generation).
- Loader: `src/data/loaders/poj104.py`, `src/data/loaders/codenet_python.py`
  chuẩn hóa về `CodeSample(sample_id, code, problem_id, language, split)`.

[🔝 Quay lại mục lục](#top)

---

## 2. Audit bắt buộc

Script: `python scripts/audit_dataset.py --dataset {poj104|codenet_python}`.
Nội dung: schema, số mẫu, số problem, phân bố problem, missing code,
exact/normalized duplicate (`src/data/validators/schema.py`,
`src/data/deduplication/duplicates.py`). Kết quả ghi vào
`reports/002_dataset_audit_poj104.md` và `reports/003_dataset_audit_codenet_python.md`.

[🔝 Quay lại mục lục](#top)

---

## 3. Cleaning & dedup

- Cleaning (`src/data/cleaning/cleaner.py`): loại empty/too_short/
  missing_problem/duplicate_submission_id/language_mismatch, mỗi record có
  reason; mặc định không lọc status (chờ DEC-012).
- Normalization (`src/data/normalization/normalizer.py`): line-ending,
  trailing whitespace, Unicode NFC — chỉ phục vụ dedup/leakage, giữ raw,
  không rename identifier.
- Dedup (`src/data/deduplication/groups.py`): exact + normalized groups,
  canonical = sample_id nhỏ nhất, mapping `duplicate_id → canonical_id`.
  Chạy **trước** split (DEC-010). POJ-104: loại cả hai loại (DEC-013).

[🔝 Quay lại mục lục](#top)

---

## 4. Split

- Ưu tiên split chính thức của dataset nếu không gây leakage cùng problem.
- Ngược lại dùng `split_by_problem` (seed 42, 80/10/10): problem ở test không
  xuất hiện ở train/val. Config: `configs/datasets/*.yaml`.
- **POJ-104: bắt buộc problem split** (official split fail leakage, DEC-007).
- CodeNet (4.4M mẫu): audit/split bằng streaming + mẫu lớn (DEC-008).

[🔝 Quay lại mục lục](#top)

---

## 5. Pair generation

- Positive = cùng problem (sau dedup nên không còn cặp copy); negative =
  khác problem, sampling theo problem, tỉ lệ 1:1 (DEC-011).
- Cap positive/problem (train/val-test riêng từng dataset,
  `configs/pairs/*.yaml`); không oversample test.
- Problems thiếu mẫu bị loại có reason (`filter_eligible`).
- Implement: `src/data/pairs/generator.py`; build: `scripts/build_dataset.py`.
- Output: `data/splits/`, `data/pairs/*.jsonl`, `data/processed/`,
  `data/manifests/`. Pair ID vô thứ tự (A,B ≡ B,A).
- Cặp xuyên split bị cấm và được kiểm tra ở `docs/leakage_prevention.md`.

[🔝 Quay lại mục lục](#top)

---

## 6. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
