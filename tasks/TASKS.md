# Kế hoạch công việc — Semantic Code Clone Detection (BigCloneBench Java)

> **Mục tiêu tuần (27-10 → 30-10-2026):** Chuyển đổi toàn bộ kiến trúc và quy trình thử nghiệm từ dataset **POJ-104 (C/C++)** sang **BigCloneBench Java** (`google/code_x_glue_cc_clone_detection_big_clone_bench`), thích ứng 6 phương pháp phát hiện clone (B1, M1–M5), hoàn thiện pipeline đánh giá F1 / Clone Types và báo cáo tổng hợp.

---

## 1. Thứ 3 (27-10-2026): Tích hợp Dataset BigCloneBench & Chuẩn bị dữ liệu

### [BCB-01] Tích hợp Loader & Schema cho BigCloneBench Java
- **Trạng thái:** Done
- **Mục tiêu:** Xây dựng module nạp dữ liệu chuẩn từ Hugging Face dataset `google/code_x_glue_cc_clone_detection_big_clone_bench`.
- **Mô tả chi tiết:**
  - Viết `src/data/loaders/bigclonebench.py` sử dụng thư viện `datasets` của Hugging Face.
  - Xử lý cấu trúc dữ liệu cặp nhị phân có sẵn: `id` (int), `id1` (int), `id2` (int), `func1` (str), `func2` (str), `label` (bool/int).
  - Tích hợp cơ chế cache local dạng Parquet/Arrow tại `data/raw/bigclonebench/` để phục vụ chạy lặp lại offline mà không phải tải lại từ web.
  - Tiền xử lý văn bản cơ bản: strip whitespace, chuẩn hóa ký tự xuống dòng `\r\n` $\to$ `\n`, loại bỏ các đoạn code rỗng hoặc lỗi decode encoding.
- **Tiêu chí hoàn thành (DoD):** Unit test `tests/test_bigclonebench_loader.py` pass; nạp thành công split mẫu train/val/test với đầy đủ các trường dữ liệu.

### [BCB-02] Thiết kế phân tập mẫu (Smoke/Dev/Full) & Audit dữ liệu Java
- **Trạng thái:** Done
- **Mục tiêu:** Tạo các phân tập con chuẩn hóa phục vụ kiểm thử nhanh và kiểm soát chất lượng dữ liệu.
- **Mô tả chi tiết:**
  - Tập gốc có tới **901.028 cặp train** và **415.416 cặp val/test** (~1,73 triệu cặp tổng cộng). Việc huấn luyện toàn bộ đòi hỏi tài nguyên lớn, do đó cần chia thành 3 cấp độ:
    - `bcb_smoke`: 500 cặp (cân bằng 50:50) cho CI/CD và smoke test luồng code tại local.
    - `bcb_dev`: 20.000 cặp (cân bằng 50:50) phục vụ validation nhanh, chốt ngưỡng và tinh chỉnh siêu tham số.
    - `bcb_full`: Toàn bộ tập dữ liệu gốc dùng khi huấn luyện trên cụm GPU Kaggle.
  - Viết script `scripts/audit_bcb_dataset.py`: thống kê phân phối độ dài dòng code, số token (BPE / AST node), tỷ lệ nhãn clone/non-clone.
  - Viết script `scripts/check_bcb_leakage.py`: kiểm tra nghiêm ngặt nguy cơ rò rỉ (leakage) giữa train và validation/test thông qua cặp trùng lặp hoặc trùng lặp hàm mã nguồn (`id1`, `id2`).
- **Tiêu chí hoàn thành (DoD):** Tạo các file manifest cố định hạt giống (fixed seed); script audit chạy không lỗi và sinh báo cáo thống kê phân phối `reports/bcb_data_audit.md`.

---

## 2. Thứ 4 (28-10-2026): Chuyển đổi Biểu diễn Mã nguồn Java (AST & CPG) & Baseline B1

### [BCB-03] Chuyển đổi bộ trích xuất AST cho Java (tree-sitter-java) cho ASTNN (M3)
- **Trạng thái:** Done
- **Mục tiêu:** Cập nhật module trích xuất cây cú pháp trừu tượng (AST) phục vụ mạng nơ-ron cây ASTNN cho ngôn ngữ Java.
- **Mô tả chi tiết:**
  - Tích hợp grammar `tree-sitter-java` vào `src/representations/ast/extract.py`.
  - Định nghĩa tập hợp các statement type chuẩn của Java: `method_declaration`, `try_with_resources_statement`, `synchronized_statement`, `catch_clause`, `enhanced_for_statement`, `lambda_expression`, `assert_statement`, v.v.
  - Áp dụng nguyên tắc trừu tượng hóa (statement abstraction của Zhang et al., 2019):
    - Node lá đưa về 3 lớp trừu tượng: `ID` (tên biến, hàm), `NUM` (số), `STR` (chuỗi ký tự).
    - Node con bên trong statement bị nén phẳng (flatten) gắn vào statement cha.
  - Sinh mảng numpy phẳng lưu cấu trúc cây để tăng tốc quá trình đóng lô (batch collation) cho PyTorch.
- **Tiêu chí hoàn thành (DoD):** Trích xuất AST thành công trên tập mẫu Java; unit test `tests/test_ast_extract_java.py` đạt 100% pass.

### [BCB-04] Chuyển đổi pipeline sinh CPG Java bằng Joern cho CPG-GNN (M4) & Hybrid (M5)
- **Trạng thái:** Done
- **Mục tiêu:** Tạo pipeline sinh Code Property Graph (CPG) cho các hàm Java bằng Joern CLI.
- **Mô tả chi tiết:**
  - Joern hỗ trợ Java gốc rất mạnh thông qua frontend `javasrc2cpg` (vượt trội hơn C/C++ do không bị lỗi thiếu include header).
  - Cập nhật script `src/representations/cpg/ingest.py` để xuất đồ thị CPG đa cạnh (AST edge, Control Flow Graph - CFG, Data Dependency Graph - DDG/CDG) cho các hàm `.java`.
  - Chuẩn hóa schema đồ thị tương thích với PyTorch Geometric: định nghĩa feature ma trận node (loại node, code token) và tensor ma trận kề cạnh theo từng loại quan hệ.
- **Tiêu chí hoàn thành (DoD):** Chạy thử nghiệm trích xuất 200 đồ thị Java mẫu; nạp thành công vào lớp `Data` của PyTorch Geometric mà không gặp xung đột kiểu dữ liệu.

### [BCB-05] Thích ứng Baseline B1 (TF-IDF + Cosine) cho Java
- **Trạng thái:** Cancelled (Không cần thiết — Baseline dùng trực tiếp dịch vụ Stanford MOSS gốc qua API/service với `language="java"`, không cần huấn luyện mô hình TF-IDF)
- **Mục tiêu:** Xây dựng mô hình Baseline truyền thống dựa trên tần suất từ khóa Java và độ tương đồng Cosine.
- **Mô tả chi tiết:**
  - Cập nhật bộ tiền xử lý từ khóa: phân tách định danh dạng camelCase (`calculateSimilarity` $\to$ `calculate`, `Similarity`), lọc bỏ các từ khóa Java thông dụng hoặc giữ lại theo n-gram.
  - Fit mô hình TF-IDF trên tập `bcb_train`; dự đoán điểm Cosine Similarity trên `bcb_val`.
  - Quét tìm ngưỡng tối ưu (threshold search) tối đa hóa F1 trên validation split và lưu vào `results/bcb_baseline_tfidf/threshold.json`.
- **Tiêu chí hoàn thành (DoD):** Script đánh giá sinh đầy đủ metrics Precision, Recall, F1 trên `bcb_dev`; ghi nhận baseline điểm số ban đầu.

---

## 3. Thứ 5 (29-10-2026): Thích ứng Mô hình Transformer & Đóng gói Kaggle Notebooks

### [BCB-06] Thích ứng Transformer cặp (CodeBERT - M1 & GraphCodeBERT - M2) cho Java
- **Trạng thái:** Done
- **Mục tiêu:** Thích ứng mô hình ngôn ngữ lớn tiền huấn luyện cho bài toán phân loại nhị phân cặp hàm Java.
- **Mô tả chi tiết:**
  - Sử dụng pretrained weights `microsoft/codebert-base` và `microsoft/graphcodebert-base` (vốn đã được pretrain sẵn trên dữ liệu Java thuộc tập CodeSearchNet).
  - Định dạng chuỗi đầu vào theo tiêu chuẩn cặp câu: `[CLS] func1 [SEP] func2 [SEP]`.
  - Cập nhật `src/models/transformer_pair.py`: cấu trúc phân loại nhị phân (Binary Classification Head) sử dụng BCEWithLogitsLoss hoặc CrossEntropyLoss 2 lớp.
  - Xây dựng 2 file cấu hình thực nghiệm: `configs/exp002_codebert_bcb.yaml` và `configs/exp003_graphcodebert_bcb.yaml`.
  - Kiểm tra 5 chế độ: `preflight` $\to$ `smoke` $\to$ `diagnostic` $\to$ `train` $\to$ `evaluate` trên môi trường local.
- **Tiêu chí hoàn thành (DoD):** Chạy chế độ `smoke` (1 batch) thành công trên local, loss hội tụ, không phát sinh lỗi tràn VRAM.

### [BCB-07] Thích ứng ASTNN (M3), CPG-GNN (M4) & Hybrid (M5) cho Java pairs
- **Trạng thái:** Done
- **Mục tiêu:** Hoàn thiện pipeline mô hình cấu trúc cây và đồ thị cho dữ liệu cặp BigCloneBench.
- **Mô tả chi tiết:**
  - **M3 (ASTNN):** Nhận 2 cây AST của cặp hàm `(func1, func2)` qua khối Dynamic Bi-GRU + Pooling, tính vector khoảng cách $v = |h_1 - h_2|$ và đưa qua tầng Dense phân loại.
  - **M4 (CPG-GNN):** Nhận 2 đồ thị CPG của cặp hàm qua kiến trúc GAT (Graph Attention Network), tổng hợp embedding toàn đồ thị và phân loại.
  - **M5 (Hybrid):** Kết hợp đặc trưng chuỗi từ GraphCodeBERT và đặc trưng đồ thị từ CPG-GNN qua cơ chế Concatenation + MLP projection.
  - Cập nhật các cấu hình: `configs/exp004_astnn_bcb.yaml`, `configs/exp005_cpg_gnn_bcb.yaml`, `configs/exp006_hybrid_bcb.yaml`.
- **Tiêu chí hoàn thành (DoD):** Chạy `smoke` pass thành công cả 3 mô hình trên tập `bcb_smoke`.

### [BCB-08] Đóng gói Kaggle Notebooks huấn luyện độc lập cho BigCloneBench
- **Trạng thái:** Done
- **Mục tiêu:** Đóng gói mã nguồn thành các Jupyter Notebooks hoàn chỉnh, tự chứa (self-contained) để huấn luyện trên GPU T4/P100 của Kaggle.
- **Mô tả chi tiết:**
  - Tạo các notebook:
    - `notebooks/bcb_exp002_codebert_kaggle.ipynb`
    - `notebooks/bcb_exp003_graphcodebert_kaggle.ipynb`
    - `notebooks/bcb_exp004_astnn_kaggle.ipynb`
    - `notebooks/bcb_exp005_cpg_extract_kaggle.ipynb`
  - Tích hợp cơ chế tự động nạp dataset từ Hugging Face qua token bí mật (secret token).
  - Tích hợp cơ chế lưu checkpoint định kỳ sau mỗi epoch (`last_checkpoint.pt`) và tự động xuất weights siêu nhẹ (`best_weights_only.pt`, chỉ ~450MB) để tiết kiệm băng thông khi kéo về máy.
  - Tích hợp cell đánh giá tự động ngay trên Kaggle sau khi huấn luyện xong để xuất file `predictions.jsonl` và `metrics.json`.
- **Tiêu chí hoàn thành (DoD):** Các notebook pass bài kiểm tra syntax và cấu trúc import; chạy thử 1 epoch trên Kaggle thành công.

---

## 4. Thứ 6 (30-10-2026): Đánh giá Tổng hợp, Custom Suite Java, Demo & Báo cáo

### [BCB-09] Xây dựng Pipeline đánh giá F1 tổng hợp & phân tích theo Clone Types (T1-T4)
- **Trạng thái:** Not started
- **Mục tiêu:** Đánh giá định lượng toàn diện 6 mô hình trên tập test của BigCloneBench và phân tích chi tiết theo loại clone.
- **Mô tả chi tiết:**
  - Tính toán các chỉ số cốt lõi: Precision, Recall, F1 tại ngưỡng tối ưu chốt từ tập Validation.
  - BigCloneBench chia thành các phân lớp clone nổi tiếng:
    - **Type-1:** Trùng lặp nguyên văn (chỉ khác khoảng trắng, chú thích).
    - **Type-2:** Trùng lặp cấu trúc, đổi tên biến/kiểu dữ liệu (syntactic clone).
    - **Strongly Type-3:** Tương đồng ngữ pháp $\ge 70\%$.
    - **Moderately Type-3:** Tương đồng ngữ pháp $50\% - 70\%$.
    - **Weakly Type-3 / Type-4:** Tương đồng ngữ pháp $< 50\%$ nhưng cùng chức năng ngữ nghĩa (semantic clone - đây là nhóm khó nhất).
  - Xây dựng ma trận phát hiện (detection rate) của từng mô hình qua từng phân lớp để chỉ ra điểm mù của TF-IDF/MOSS so với ASTNN/CodeBERT/CPG-GNN.
  - Thực hiện kiểm định ý nghĩa thống kê: McNemar test và 1000-sample Bootstrap Confidence Interval 95%.
- **Tiêu chí hoàn thành (DoD):** Sinh file tổng hợp `results/bcb_leaderboard.json` và bảng so sánh Markdown `reports/bcb_leaderboard.md`.

### [BCB-10] Chuyển đổi Custom Test Suite & cập nhật Web Demo sang Java
- **Trạng thái:** Not started
- **Mục tiêu:** Cung cấp bộ test thực tế nhỏ gọn có thể giải thích trực quan và tích hợp vào giao diện demo.
- **Mô tả chi tiết:**
  - Chuyển đổi các bài toán test trong `data/custom_suite/` (trước đây viết bằng C++) sang mã nguồn **Java chuẩn**:
    - Bài toán thao tác danh sách liên kết (`class LinkedList` vs hàm tự do).
    - Các biến thể clone: đổi vòng lặp `for` sang `while`, đổi cấu trúc điều khiển, đổi phương thức thư viện.
    - Các cặp non-clone khó: cùng khung giải thuật / skeleton nhưng khác ngữ nghĩa xử lý dữ liệu.
  - Cập nhật Web Demo (FastAPI + Vanilla JS):
    - Hỗ trợ cú pháp Java trong bộ highlight mã nguồn.
    - Tích hợp trọng số mô hình đã train trên BigCloneBench vào backend suy luận localhost.
- **Tiêu chí hoàn thành (DoD):** Chạy `python -m src.demo.app` hiển thị giao diện đối chiếu 2 đoạn mã Java và đưa ra điểm số kèm highlight trực quan.

### [BCB-11] Chốt số liệu, cập nhật Decision Log (DEC-035) & xuất Báo cáo tổng hợp
- **Trạng thái:** Not started
- **Mục tiêu:** Hoàn thiện hồ sơ tài liệu khoa học và đồng bộ tiến độ dự án.
- **Mô tả chi tiết:**
  - Viết `DEC-035: Chuyển dịch toàn bộ phạm vi sang BigCloneBench (Java)` trong `docs/decision_log.md`.
  - Cập nhật lại `README.md`, `project_structure.md`, `docs/dataset_protocol.md`.
  - Tổng hợp toàn bộ kết quả, phân tích đồ thị, so sánh ưu nhược điểm các phương pháp thành báo cáo tuần/báo cáo tổng hợp biên dịch bằng XeLaTeX (`reports/[31-10] - Báo cáo tổng hợp.tex` $\to$ PDF).
- **Tiêu chí hoàn thành (DoD):** File PDF báo cáo biên dịch sạch (0 error / 0 Overfull); toàn bộ tài liệu dự án đồng bộ 100% với BigCloneBench.
