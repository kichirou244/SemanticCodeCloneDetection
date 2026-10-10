# Giải thích luồng Highlight code giống nhau của MOSS

Phần tô sáng (highlight) đoạn code giống nhau trong project được thiết kế đi qua 3 bước (từ lúc nhận kết quả của MOSS về cho đến lúc hiển thị lên màn hình). Dưới đây là chi tiết luồng hoạt động cùng các file tương ứng:

## 1. Trích xuất khoảng ký tự (Spans) từ kết quả MOSS
**File xử lý:** [moss_service.py](file:///d:/Edu/NLP/SemanticCodeCloneDetection/src/models/baseline/moss_service.py) -> Hàm `evidence()`

- Khi quét 2 file, code không tự tìm đoạn giống nhau mà nộp lên server MOSS (Stanford) thông qua thư viện `mosspy`.
- MOSS sẽ trả về kết quả dưới dạng các trang HTML, trong đó các đoạn code bị đánh giá là trùng lặp sẽ được bọc trong thẻ `<font color="#FF0000">...</font>`.
- Hàm `extract_matched_blocks()` sẽ bóc tách (scrape) các đoạn text màu đỏ này ra khỏi HTML.
- Sau đó, hàm `locate_in_code()` sẽ lấy các đoạn text đó và dùng lệnh tìm kiếm chuỗi (`code.find()`) quét lại vào chuỗi source code ban đầu để xác định tọa độ **khoảng ký tự (character spans)**. Điều này cho biết đoạn code giống nhau bắt đầu từ ký tự thứ mấy đến ký tự thứ mấy (`[start, end)`).
- Kết quả được trả về dưới dạng danh sách các mảng ký tự `spans_a` và `spans_b`.

## 2. Quy đổi Ký tự sang Token Index (Chỉ số Token)
**File xử lý:** [service.py](file:///d:/Edu/NLP/SemanticCodeCloneDetection/src/demo/service.py) -> Các hàm `_spans_to_indices()` và `_attach_token_indices()`

- Ở Frontend, code không được render thành một khối text duy nhất mà được cắt nhỏ ra thành các từ khóa (tokens) để hỗ trợ tô màu cú pháp. Do đó, tọa độ ký tự lấy từ bước 1 không thể áp dụng trực tiếp được lên giao diện.
- Hàm `_spans_to_indices()` ở backend sẽ làm nhiệm vụ đồng bộ: Nó duyệt qua từng token của file code (đã được lexer phân tích). Nếu tọa độ khoảng ký tự của token đó có giao cắt (nằm đè lên / overlap) với các đoạn `spans` của MOSS tìm được, thì token đó sẽ được đánh dấu là thuộc phần copy.
- Quá trình này tạo ra 2 mảng phẳng là `tok_a` và `tok_b` (chứa các số thứ tự index của những token cần được highlight, ví dụ: `[5, 6, 7, 12, ...]`). Hàm `_attach_token_indices()` sẽ đóng gói các index này vào payload trả về cho Frontend.

## 3. Frontend hiển thị Highlight
**File xử lý:** [diffPane.js](file:///d:/Edu/NLP/SemanticCodeCloneDetection/src/demo/web/js/components/diffPane.js) và [main.js](file:///d:/Edu/NLP/SemanticCodeCloneDetection/src/demo/web/js/main.js)

- Trong file `diffPane.js`, khi xây dựng giao diện hiển thị code, mỗi thẻ `<span>` chứa token đều được gắn thêm một biến dataset `data-ti` tương ứng với chỉ số index của token đó. (Ví dụ: `<span class="tok" data-ti="5">...</span>`).
- Trong file `main.js` (tại hàm `applyEvidence()`), khi bạn click chọn xem evidence của MOSS, code sẽ lấy danh sách `tok_a` và `tok_b` từ API payload.
- Logic sau đó sẽ dò tìm tất cả các thẻ `<span data-ti="...">` có index nằm trong danh sách trên và gán thêm cho nó class CSS là `.tok-ev`.
- Class `.tok-ev` này được định nghĩa trong CSS để đổi màu nền (background) thành màu vàng, tạo ra hiệu ứng highlight trên trình duyệt.

---

## Tại sao phần Highlight đôi khi nhìn "chưa được ổn lắm"?

Phần này đôi khi bị lệch là do các hạn chế kỹ thuật tất yếu của phương pháp móc nối (mapping) text thuần từ MOSS:

1. **Sai số khi dò tìm lại chuỗi (String Matching):**
   MOSS thường nhét thêm các dòng rỗng (artifact) hoặc bỏ qua comment. Do đó, đoạn text bóc ra từ kết quả MOSS không khớp 100% về khoảng trắng/comment so với file gốc. Hàm `locate_in_code()` đành phải dùng thuật toán tìm chuỗi thô (như `code.find()`). Khi trong một file có nhiều đoạn code lặp lại giống nhau (như boilerplate, vòng lặp chuẩn), nó có thể match nhầm vào đoạn xuất hiện trước thay vì đoạn thực sự bị copy.
   
2. **Sai lệch do làm tròn Token (Token Overlap):**
   Giao diện bôi màu theo đơn vị Token, nhưng MOSS trả về theo đơn vị Ký tự. Chỉ cần một đoạn chữ bị bôi đỏ của MOSS vô tình chạm/cắt qua 1 phần của một Token ở bước 2, thì nguyên cả Token đó sẽ bị bôi vàng. Việc này khiến cho vệt highlight đôi khi dài hơn thực tế (thừa ra ở hai đầu).

3. **MOSS không ánh xạ cặp (No Pair Mapping):**
   MOSS chỉ xác định "File này bị copy những vùng này", chứ không trả về bản đồ ánh xạ 1-1 chính xác kiểu "Cụm token X ở File A chính là cụm token Y ở File B" (như khái niệm `tok_pairs` của các mô hình Deep Learning sau này). Vì không có luồng kết nối rõ ràng, Frontend chỉ có thể bôi chung một màu vàng nguyên khối cho tất cả thay vì nối đường chỉ phân biệt rõ nét từng cụm.
