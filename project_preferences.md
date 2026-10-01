# Project Preferences

| Mục | Nội dung |
| :--- | :--- |
| **Tên dự án** | Semantic Code Clone Detection — Web Demo (semantic code clone detection demo) |
| **Loại dự án** | Web App (demo nghiên cứu nội bộ, chạy localhost) |
| **Ngày lưu lựa chọn** | 2026-09-28 |
| **Trạng thái** | Approved |

> Phạm vi: công cụ trình diễn phục vụ nghiên cứu, chỉ chạy trên máy cá nhân
> (`127.0.0.1`), không phải website tiếp thị. Vì vậy các hạng mục SEO /
> chuyển đổi / uy tín bán hàng không có ý nghĩa và được bỏ qua có chủ đích.

## Checklist Chuyển đổi Web/App

### Đã chọn triển khai
- [ ] Không mục nào

### Không chọn
- [ ] 1. Custom 404 Page
- [ ] 2. CTA Above the Fold
- [ ] 3. Internal Links
- [ ] 4. Thank You Page
- [ ] 5. Case Studies
- [ ] 6. 5 FAQ's
- [ ] 7. Response Time Promise
- [ ] 8. Sticky Mobile CTA
- [ ] 9. robots.txt
- [ ] 10. Unique Page Titles
- [ ] 11. Meta Description
- [ ] 12. Maps + Directions
- [ ] 13. Real Reviews
- [ ] 14. Alt Text on Images
- [ ] 15. Local Schema
- [ ] 16. Privacy Policy Page
- [ ] 17. Google Analytics
- [ ] 18. Team Photo

## Quyết định kỹ thuật đã chốt (2026-09-28)

| Hạng mục | Lựa chọn | Lý do |
| :--- | :--- | :--- |
| Stack web | FastAPI + uvicorn + HTML/JS/CSS thuần (ES modules) | Thêm đúng 2 dependency. Cần toàn quyền render highlight đỏ/xanh từng token, Gradio làm không nổi. Không CDN ⇒ chạy offline. |
| Cơ chế highlight | So khớp token (difflib LCS) | Nhanh, tất định, không GPU, tái lập được. |
| Chế độ căn chỉnh | `literal` (nguyên văn) + `abstract` (gộp mọi identifier thành `ID`) | Chế độ `abstract` tái hiện đúng lớp trừu tượng của ASTNN, cho thấy vì sao model semantic thắng baseline từ khoá. |
| Phạm vi mô hình | B1, M1, M2, M3 | M4 chưa có checkpoint (blocked Joern), M5 là stub. Demo chỉ hiện model thật. |
| Ngưỡng quyết định | Đọc từ `results/*/threshold.json` (đã chọn trên validation) | Không hardcode. Demo phản ánh đúng protocol đánh giá của EXP-001..004. |

## Lịch sử thay đổi

| Phiên bản | Ngày thay đổi | Tác giả | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1.0.0 | 2026-09-28 | AI Agent | Khởi tạo. Người dùng chọn **không triển khai mục nào** trong 18 hạng mục chuyển đổi web/app (demo nghiên cứu localhost). Chốt stack FastAPI + vanilla JS và cơ chế highlight bằng so khớp token. |
