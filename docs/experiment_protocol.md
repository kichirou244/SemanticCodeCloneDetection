<a id="top"></a>

# Experiment protocol (experiment_protocol.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.2.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Fairness](#1-fairness)
- [2. Cấu hình](#2-cau-hinh)
- [3. Tracking](#3-tracking)
- [4. Registry & trạng thái](#4-registry--trang-thai)
- [5. Pair architecture chuẩn (EXP-002)](#5-pair-architecture-chuan-exp-002)
- [6. Lịch sử thay đổi](#6-lich-su-thay-doi)

---

## 1. Fairness

Mọi model (B1, M1–M5) chạy trên cùng split, cùng protocol pair generation, cùng
seed và cùng metric chính (Precision/Recall/F1). Retrieval (nếu dùng) có metric
riêng, không trộn với classification.

[🔝 Quay lại mục lục](#top)

---

## 2. Cấu hình

Không hard-code: mọi siêu tham số nằm ở `configs/datasets/*.yaml` và
`configs/models/*.yaml`, nạp qua `src/common/config.py`. Seed mặc định 42
(`src/common/seed.py`).

[🔝 Quay lại mục lục](#top)

---

## 3. Tracking

Mỗi experiment lưu: experiment_id, dataset + version, language, model + config,
seed, thời gian train, best epoch/val metric, test metrics, checkpoint path,
git commit, môi trường (`scripts/env_report.py`). Kết quả generate từ pipeline,
không chép tay.

[🔝 Quay lại mục lục](#top)

---

## 4. Registry & trạng thái

Theo dõi ở `docs/experiment_registry.md` với trạng thái
PLANNED/RUNNING/COMPLETED/FAILED/BLOCKED/INVALIDATED/READY_FOR_GPU.
Experiment INVALIDATED giữ report và ghi lý do.

[🔝 Quay lại mục lục](#top)

---

## 5. Pair architecture chuẩn (EXP-002)

M1 dùng encoder shared (siamese): `pair = [hA ; hB ; |hA-hB|]` →
MLP (Tanh, dropout) → logit. Threshold trên validation tối ưu F1, freeze rồi
đánh giá test 1 lần — cùng protocol với EXP-001 để so sánh công bằng
(DEC-015, DEC-016).

[🔝 Quay lại mục lục](#top)

---

## 6. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
| 2 | 1.2.0 | 2026-09-20 | Thêm pair architecture chuẩn + trạng thái READY_FOR_GPU. |
