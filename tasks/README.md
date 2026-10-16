# Đẩy task lên Notion (trang "Project Manager")

Script: `tasks/notion.js` — đọc `tasks/tasks.json`, dò database con trong trang
Notion rồi tạo từng task thành 1 page.

Trang đích: <https://app.notion.com/p/kichirou/Project-Manager-4c1d6f72e24d832ead7c01c4953cb33a>

---

## Cần cung cấp những gì

1. **`NOTION_TOKEN`** — Internal Integration Secret.
   - Vào <https://www.notion.so/my-integrations> → **New integration** → chọn workspace → copy
     **Internal Integration Secret** (dạng `secret_...` hoặc `ntn_...`).

2. **Share trang cho integration (bắt buộc, không thì lỗi 404).**
   - Mở trang **Project Manager** → góc phải **`...`** → **Connections** (hoặc *Add connections*)
     → chọn integration vừa tạo.

3. **`NOTION_PAGE_ID`** — đã có sẵn trong `.env.example`
   (`4c1d6f72e24d832ead7c01c4953cb33a`). Nếu database không nằm trong trang này,
   set thẳng `NOTION_DATABASE_ID` (lấy từ URL database, chuỗi 32 ký tự trước `?v=`).

4. **(tuỳ chọn) Tên property** — mặc định script tự dò theo kiểu dữ liệu:
   - cần 1 property **title** (bắt buộc),
   - **date** / **status** / **rich_text** thì có cũng được, không có thì bỏ qua.
   - Nếu tên khác lạ, set `NOTION_TITLE_PROP`, `NOTION_DATE_PROP`, `NOTION_STATUS_PROP`,
     `NOTION_NOTES_PROP`.

> Lưu ý: property kiểu **select/status** — task trong `tasks.json` chỉ gửi `status`
> nếu bạn điền; để trống thì Notion dùng giá trị mặc định của cột.

---

## Gắn task vào project (không cần biết id)

Không cần tự đi tìm id project. Chỉ cần **tên project** trong `tasks/.env`:

```bash
NOTION_PROJECT_NAME=[HUSC] NLP: SemanticCodeCloneDetection
```

Script sẽ tự:
1. vào db projects, tìm page có title khớp tên;
2. lấy page id đó;
3. gắn vào cột **relation** của task trong db tasks (tự dò cột, hoặc set `NOTION_PROJECT_PROP`).

Kiểm tra trước khi tạo:
```bash
node tasks/notion.js --project      # chỉ tìm + in ra project id rồi thoát
node tasks/notion.js --schema       # in cả cột relation sẽ gắn
```

Nếu tên khớp nhiều project, script in danh sách và dùng cái đầu; tên khớp chính xác
được ưu tiên.

---

## Cách chạy

```bash
# (lần đầu) điền NOTION_TOKEN vào tasks/.env

node tasks/notion.js --schema                 # xem cột + mapping (không tạo gì)
node tasks/notion.js --tasks                  # liệt kê task hiện có (ngày, status)
node tasks/notion.js --dry-run                # xem trước tasks.json

node tasks/notion.js                          # SYNC: có thì cập nhật, chưa có thì tạo (không trùng)
node tasks/notion.js --status "<tên task>" "In progress"   # đổi status 1 task
node tasks/notion.js --archive "<tên task>"   # ẩn/xoá 1 task
```

Sửa task ở `tasks/tasks.json` (mỗi task: `title`, `date` (YYYY-MM-DD), `status?`, `notes?`).
Sync khớp theo **tên task** (trùng tên tuyệt đối), nên đổi ngày/chú thích thì chỉ cần chạy lại.

### Trạng thái (Status)
`Not started` · `In progress` · `On Hold` · `To Review` · `Done` · `Cancelled`.

Quy ước: trước khi bắt đầu 1 việc → đổi sang `In progress`; xong → `Done`.


Sửa task ở `tasks/tasks.json` (mỗi task: `title`, `date` (YYYY-MM-DD), `status?`, `notes?`).

Không cần `npm install` — script dùng `fetch` sẵn của Node (cần Node ≥ 18).

---

## Nếu gặp lỗi

| Lỗi | Nguyên nhân thường gặp |
| --- | --- |
| `401 Unauthorized` | `NOTION_TOKEN` sai/thiếu. |
| `404 Not found` | Chưa share trang cho integration, hoặc sai page/database id. |
| `Không tìm thấy database con trong trang` | Trang không có database, hoặc database nằm chỗ khác → set `NOTION_DATABASE_ID`. |
| `400 ... property does not exist` | Sai tên property → set biến `NOTION_*_PROP` đúng tên, chạy `--schema` để xem. |
| `400 ... invalid status option` | Cột Status là `status`/`select` không có option bạn điền → bỏ `status` trong `tasks.json`. |
