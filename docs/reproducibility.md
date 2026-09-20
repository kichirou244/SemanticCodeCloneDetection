<a id="top"></a>

# Reproducibility (reproducibility.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Công thức tái lập](#1-cong-thuc-tai-lap)
- [2. Thứ phải lưu](#2-thu-phai-luu)
- [3. Lệnh kiểm chứng](#3-lenh-kiem-chung)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Công thức tái lập

```text
same code + same dataset version + same config + same seed + same environment
= reproducible result
```

Seed tập trung ở `src/common/seed.py`; config ở `configs/`; dataset version/hash
ghi vào report khi có.

[🔝 Quay lại mục lục](#top)

---

## 2. Thứ phải lưu

Python/PyTorch/CUDA/GPU (`scripts/env_report.py`), dependency
(`requirements.txt`), dataset version, checkpoint, seed, config, git commit.

[🔝 Quay lại mục lục](#top)

---

## 3. Lệnh kiểm chứng

```bash
python scripts/env_report.py
pytest tests/ -q
python scripts/smoke_baseline.py
```

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
