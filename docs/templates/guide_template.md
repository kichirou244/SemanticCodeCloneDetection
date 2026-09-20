<a id="top"></a>

# [Tên Hướng dẫn / Quy trình] (guide_template.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.0.0 |
| **Ngày cập nhật** | [YYYY-MM-DD] |
| **Tác giả** | [Tên tác giả] |
| **Trạng thái** | Draft / Reviewing / Approved |

---

## Mục lục

- [1. Mục đích](#1-muc-dich)
- [2. Điều kiện chuẩn bị (Prerequisites)](#2-dieu-kien-chuan-bi-prerequisites)
- [3. Các bước thực hiện (Step-by-step Steps)](#3-cac-buoc-thuc-hien-step-by-step-steps)
- [4. Xử lý sự cố thường gặp (Troubleshooting)](#4-xu-ly-su-co-thuong-gap-troubleshooting)
- [5. Lịch sử thay đổi](#5-lich-su-thay-doi)

---

## 1. Mục đích
*[Mô tả ngắn gọn mục đích của hướng dẫn này, ví dụ: Hướng dẫn cài đặt cơ sở dữ liệu Postgres trong môi trường Docker local.]*

---

## 2. Điều kiện chuẩn bị (Prerequisites)
*[Liệt kê các công cụ, phiên bản phần mềm, quyền truy cập cần thiết trước khi bắt đầu thực hiện.]*

*   Ví dụ: Docker Desktop phiên bản >= 20.10.
*   Ví dụ: Tài khoản admin trên hệ thống Database.

---

## 3. Các bước thực hiện (Step-by-step Steps)
*[Mô tả chi tiết từng bước bằng danh sách đánh số hoặc các đoạn mã lệnh cụ thể.]*

1.  **Bước 1**: Khởi chạy container docker.
    ```bash
    docker compose up -d
    ```
2.  **Bước 2**: [Mô tả chi tiết bước tiếp theo...]

---

## 4. Xử lý sự cố thường gặp (Troubleshooting)
*[Các lỗi phổ biến có thể xảy ra và cách khắc phục nhanh.]*

*   **Lỗi 1**: Cổng kết nối bị trùng (Port already in use).
    *   *Cách xử lý*: Đổi port trong file `.env` hoặc dừng dịch vụ đang chiếm cổng.
*   **Lỗi 2**: [Lỗi khác và cách xử lý...]

---

## 5. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | [YYYY-MM-DD] | Khởi tạo tài liệu hướng dẫn. |
