<a id="top"></a>

# Bản đồ Cấu trúc Thư mục Tài liệu & Hướng dẫn Tra cứu (directory_map.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.3.0 |
| **Ngày cập nhật** | 2026-08-28 |
| **Tác giả** | Phát triển Hệ thống |
| **Trạng thái** | Xác nhận |

---

## Mục lục

- [1. Mục đích](#1-muc-dich)
- [2. Cấu trúc Thư mục Tài liệu (docs/)](#2-cau-truc-thu-muc-tai-lieu-docs)
- [3. Hướng dẫn Tìm kiếm và Tra cứu cho AI Agent](#3-huong-dan-tim-kiem-va-tra-cuu-cho-ai-agent)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Mục đích

Tài liệu này đóng vai trò như một "bản đồ" cấu trúc thư mục tài liệu trong dự án, giúp nhà phát triển và các AI Agent định vị thông tin cần thiết một cách nhanh nhất, giảm thiểu số lần tìm kiếm mù quáng bằng `grep` và tối ưu hóa tài nguyên.

---

## 2. Cấu trúc Thư mục Tài liệu (docs/) & Kỹ năng Agent (.agents/)

Toàn bộ tài liệu kỹ thuật phải được phân loại và lưu trữ tại các thư mục con tương ứng bên dưới thư mục `docs/`. Tuyệt đối không tạo tài liệu tự do trực tiếp trong thư mục gốc `docs/`. Các cấu hình và bộ kỹ năng định hình hành vi của AI Agent được quản lý tại `.agents/`.

*   **`docs/specs/` (Đặc tả thiết kế & Yêu cầu)**:
    *   **`system/` (Hệ thống chung)**:
        *   `requirements.md`: Đặc tả yêu cầu hệ thống (PRD tổng quan).
        *   `document_management_policy.md`: Quy chế quản lý và cấu trúc tài liệu bắt buộc.
        *   `directory_map.md`: Bản đồ cấu trúc thư mục tài liệu & hướng dẫn tra cứu.
    *   **`features/` (Đặc tả tính năng)**:
        *   `[tên_tính_năng]/[đặc_tả_tính_năng].md`: Đặc tả chi tiết từng tính năng nghiệp vụ.
    *   **`architecture/` (Kiến trúc & Hạ tầng)**:
        *   `[đặc_tả_kiến_trúc].md`: Đặc tả kiến trúc kỹ thuật/hạ tầng.
    *   **`api/` (Tài liệu API)**:
        *   Chứa danh sách API (`api_list.md`), đặc tả REST API (`docs/specs/api/rest/`) và Realtime API (`docs/specs/api/realtime/`).
*   **`docs/guides/` (Hướng dẫn & Quy trình)**:
    *   Chứa hướng dẫn thiết lập môi trường phát triển, cài đặt cơ sở dữ liệu, các bước chạy lệnh deploy, chạy test, quy trình onboarding cho nhà phát triển mới.
*   **`docs/templates/` (Biểu mẫu tài liệu)**:
    *   Nơi lưu trữ các file Markdown mẫu như: Đặc tả tính năng, REST API, Realtime API, Hướng dẫn (Guides) và Quyết định kiến trúc (Proposals/ADR).
*   **`docs/features/` (Ghi chú tính năng & Tiến độ)**:
    *   Chứa thông tin tóm tắt tiến độ, các checklist thực tế của từng tính năng đang được phát triển (State/Gap list so với đặc tả chính thức).
*   **`docs/analysis/` (Báo cáo & Phân tích)**:
    *   Chứa báo cáo kiểm tra hiệu suất, phân tích kiến trúc, phân tích lỗ hổng bảo mật hoặc nghiên cứu kỹ thuật trước khi làm tính năng.
*   **`docs/proposals/` (Đề xuất kỹ thuật)**:
    *   Lưu trữ các đề xuất nâng cấp hệ thống, đề xuất thay đổi công nghệ hoặc thay đổi kiến trúc (ADR - Architecture Decision Records).
*   **`docs/migration/` (Di chuyển & Nâng cấp)**:
    *   Chứa các kế hoạch di chuyển cơ sở dữ liệu (migration), chuyển đổi phiên bản framework hoặc nhật ký chuyển đổi hệ thống cũ sang hệ thống mới.
*   **`docs/assets/` (Tài nguyên hình ảnh & Sơ đồ)**:
    *   Nơi chứa các sơ đồ diagram, file ảnh đính kèm trong tài liệu.
*   **`.agents/skills/` (Quy chuẩn kỹ năng AI Agent)**:
    *   `development_protocol/SKILL.MD`: Quy định quy trình phát triển, quy tắc an toàn chạy lệnh và các chuẩn mực code bắt buộc (Clean Code & SOLID, tư duy Senior, phân rã component/hook/service, quản lý state & global state, cấu trúc folder/project chuyên nghiệp).
    *   `documentation/SKILL.MD`: Quy định quy chuẩn viết tài liệu Markdown, quản lý phiên bản tài liệu và cấu trúc tài liệu dự án.
    *   `frontend_protocol/SKILL.MD`: Quy định tiêu chuẩn thiết kế UI/UX cao cấp, phát triển giao diện tương tác và tối ưu hóa phía Client (phân rã component, custom hooks, state/global state, cấu trúc folder Feature-Based).
    *   `backend_protocol/SKILL.MD`: Quy chuẩn thiết kế kiến trúc hệ thống backend, tối ưu hóa CSDL, bảo mật thông tin và viết API chuẩn hóa (phân tầng Controller/Service/Repository/Domain, cấu trúc folder Module-Based).
    *   `ai_model_protocol/SKILL.MD`: Hướng dẫn quy trình chuẩn hóa xây dựng, huấn luyện, đánh giá và triển khai mô hình AI/ML.
    *   `web_conversion_protocol/SKILL.MD`: Quy định checklist chuyển đổi web/app bắt buộc hỏi người dùng khi khởi đầu dự án (Custom 404 Page, CTA Above the Fold, SEO, Trust,...) và lưu lựa chọn vào tệp `project_preferences.md`.

*   **`project_preferences.md` (Thư mục gốc dự án)**: Tệp lưu lựa chọn checklist chuyển đổi web/app của người dùng theo quy định tại `development_protocol/SKILL.MD` và `web_conversion_protocol/SKILL.MD`. Tệp này đặt tại thư mục gốc dự án (`project_preferences.md`), không nằm trong `docs/`.

---

## 3. Hướng dẫn Tìm kiếm và Tra cứu cho AI Agent

Khi nhận được yêu cầu phát triển hoặc bảo trì tính năng, AI Agent phải tuân thủ nghiêm ngặt quy trình tra cứu thông tin sau đây:

1.  **Đọc Bản đồ Thư mục**: Xem `docs/specs/system/directory_map.md` đầu tiên để hiểu cấu trúc dự án.
2.  **Đọc Chính sách Quản lý**: Xem `docs/specs/system/document_management_policy.md` để hiểu quy định viết/sửa đổi tài liệu và thứ tự ưu tiên.
3.  **Xác định Tài liệu Nghiệp vụ**: Đọc tài liệu yêu cầu hệ thống cấp cao nhất (`docs/specs/system/requirements.md`) liên quan đến tính năng đó.
4.  **Kiểm tra Đặc tả Chi tiết**: Kiểm tra các tệp đặc tả kỹ thuật cụ thể nằm trong các thư mục con tương ứng (ví dụ: `docs/specs/features/`, `docs/specs/architecture/` hoặc `docs/specs/api/`) trước khi xem mã nguồn.
5.  **Chỉ tìm kiếm có khoanh vùng**: Tránh sử dụng lệnh `grep` tìm kiếm trên diện rộng trên toàn bộ thư mục dự án khi chưa xác định được vùng code. Chỉ tìm kiếm tập trung vào các folder chức năng.

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-08-07 | Khởi tạo tài liệu bản đồ cấu trúc thư mục dự án chuẩn hóa bằng tiếng Việt. |
| 2 | 1.1.0 | 2026-08-07 | Cập nhật cấu trúc thư mục kỹ năng mới của Agent (.agents/skills) bao gồm FE, BE và AI. |
| 3 | 1.2.0 | 2026-08-28 | Bổ sung kỹ năng `web_conversion_protocol` (checklist chuyển đổi web/app) và tệp `project_preferences.md` lưu lựa chọn của người dùng ở thư mục gốc dự án. |
| 4 | 1.3.0 | 2026-08-28 | Cập nhật mô tả chi tiết các kỹ năng `development_protocol`, `frontend_protocol` và `backend_protocol` theo chuẩn code Senior (phân rã component/hook, state & global state, cấu trúc folder Feature/Module-Based). |