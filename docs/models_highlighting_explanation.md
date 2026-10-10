# Tóm tắt thuật toán Highlight của 5 mô hình Deep Learning (M1 - M5)

Khác với MOSS dùng so khớp chuỗi (string matching), 5 phương pháp DL sử dụng kỹ thuật "Explainable AI" (trích xuất vùng mô hình chú ý) với các phép toán cụ thể sau:

---

## 1. M1 và M2 (CodeBERT, GraphCodeBERT)
*Áp dụng trên nhánh Text.*

* **Tô màu (Saliency Map):** Tính đạo hàm ngược (backward pass) từ điểm dự đoán cuối (logit) về lớp vector đầu vào (embeddings). 
  * **Công thức:** `Saliency = abs(gradient * embedding)`. 
  * Các token có điểm Saliency cao hơn mức trung bình sẽ được tô màu.
* **Bắt cặp:** Tính độ tương đồng **Cosine Similarity** giữa vector nhúng của từng từ ở hai file để tìm cặp giống nhau nhất chéo qua lại (mutual best).

## 2. M3 (ASTNN)
*Làm việc với câu lệnh (statement) trên cây AST.*

* **Toán học:** Lấy vector biểu diễn (RvNN) của từng câu lệnh ở file A và file B, tính **Cosine Similarity** để tìm cặp câu lệnh giống nhau nhất chéo qua lại (mutual best).
* **Quy tắc lọc để bôi màu:**
  1. Loại bỏ khối lệnh có span bao phủ > 60% toàn bộ file.
  2. Nếu lệnh to bọc lệnh nhỏ (ví dụ `for` bọc `if`) và cả hai đều được bắt cặp, chỉ bôi màu lệnh lõi ở trong (`if`).

## 3. M4 (CPG-GNN)
*Làm việc với đồ thị mã (CPG) qua mạng GAT.*

* **Tô màu (GAT Attention):** Trích xuất **trọng số Attention** từ lớp mạng GAT cuối cùng. Cộng dồn trọng số này cho các node (nút) thuộc cùng một dòng code. Chọn ra **5 dòng code** có tổng trọng số lớn nhất để tô màu.
* **Bắt cặp:** Tính vector đại diện cho một dòng code bằng cách lấy trung bình vector của các node trên dòng đó. Tính **Cosine Similarity** để chọn ra tối đa 10 cặp dòng giống nhau nhất (mutual best).

## 4. M5 (Hybrid GraphCodeBERT + CPG-GNN)
*Kết hợp mô hình ngôn ngữ (Text) và mạng đồ thị (Graph).*

* **Toán học (Saliency Map):** Mặc dù chấm điểm bằng cả Text và Graph, mô hình **chỉ giữ đạo hàm lùi trên nhánh Text**, đóng băng nhánh Graph. Công thức Saliency tương tự M1/M2: `abs(gradient * embedding)`.
* **Bắt cặp:** Tương tự M1/M2, lấy vector nhúng của phần Text và dùng **Cosine Similarity** để tìm từ khóa tương đồng nhất (mutual best).
