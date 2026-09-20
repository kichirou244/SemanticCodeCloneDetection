<a id="top"></a>

# Leakage prevention (leakage_prevention.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.1.1 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Nguyên tắc](#1-nguyen-tac)
- [2. Các kiểm tra](#2-cac-kiem-tra)
- [3. Quy trình](#3-quy-trinh)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Nguyên tắc

- Chia train/val/test **trước** augmentation/transformation.
- Không để cùng problem, cùng source (exact/normalized), cặp clone, hay bản
  transform của cùng source rơi vào hai split khác nhau khi đánh giá
  generalization.

[🔝 Quay lại mục lục](#top)

---

## 2. Các kiểm tra

`src/data/leakage/checks.py` (`run_pair_leakage_audit`, Phase 1):

1. Problem overlap xuyên split (= 0 mới đạt).
2. Sample ID overlap xuyên split (= 0).
3. Code overlap exact xuyên split (= 0). Định nghĩa: byte-identical
   sha256 của raw code (không collapse whitespace).
4. Code overlap normalized xuyên split (= 0). Định nghĩa duy nhất:
   sha256 của `CodeNormalizer` output (bảo thủ — giữ whitespace nội bộ vì
   indent Python mang semantics; xem report 008).
5. Pair overlap vô thứ tự (A,B ≡ B,A) xuyên split (= 0).
6. (Phase sau) near-duplicate, augmentation leakage.

`build_dataset.py` tự chạy audit cuối và **dừng (exit 1) nếu fail** —
dataset fail không được sử dụng.

[🔝 Quay lại mục lục](#top)

---

## 3. Quy trình

1. Chạy `scripts/audit_dataset.py` → chốt split.
2. Chạy `scripts/check_leakage.py` → ghi `reports/004_leakage_audit.md`.
3. Experiment nào fail leakage bị đánh dấu `INVALIDATED`, giữ report, không xóa.
4. Mọi ngoại lệ phải ghi decision log với lý do.

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
| 2 | 1.1.0 | 2026-09-20 | Thêm kiểm tra sample/normalized/pair-level + fail-fast ở build. |
| 3 | 1.1.1 | 2026-09-20 | Sửa exact thành byte-identical (bỏ collapse whitespace nội bộ). |
