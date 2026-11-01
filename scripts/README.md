# Danh mục Kịch bản Thực thi (Scripts Reference)

Thư mục `scripts/` chứa toàn bộ các kịch bản thực thi phục vụ quy trình thu thập dữ liệu, trích xuất đặc trưng, sinh notebook huấn luyện và đánh giá thực nghiệm cho đề tài.

---

## 1. Đánh giá & Benchmark (Evaluation & Metrics)

Các script trọng tâm dùng để tái hiện kết quả trên các tập dữ liệu:

- **`evaluate_bcb_5000_test.py`**: Đánh giá 10 biến thể mô hình (M1, M2, M3, M4 với các cấp độ Chưa huấn luyện, Đóng băng, Fine-tuning) trên đúng **5.000 cặp Test độc lập của BigCloneBench**. Kết quả ghi nhận tại `results/bcb_test_5k_results.json`.
- **`evaluate_coder_benchmark.py`**: Đánh giá khả năng chuyển dịch chéo ngôn ngữ (Cross-language Zero-shot) trên **400 cặp bài nộp C/C++ sinh viên từ CODER HUSC**, phân tích chi tiết độ chính xác theo từng loại trùng lặp: Type-1/2/ST3, MT3, Type-4, Non-clone.
- **`score_custom_suite_java.py`**: Chấm điểm các mô hình trên tập kiểm thử phân tầng Java nội bộ.
- **`score_custom_suite_cpp.py`**: Chấm điểm các mô hình trên tập kiểm thử C/C++ mở rộng.
- **`eval_moss_service.py` / `eval_moss.py`**: Đo lường và đối sánh kết quả với hệ thống phát hiện gian lận MOSS (Stanford University).

---

## 2. Thu thập & Xây dựng Dữ liệu (Data Preparation)

- **`crawl_coder_husc.py`**: Thu thập dữ liệu bài nộp sinh viên C/C++ từ hệ thống chấm tự động CODER HUSC.
- **`prepare_coder_benchmark.py`**: Chuẩn hóa, lọc mã nguồn và tạo bộ dữ liệu 400 cặp kiểm thử thực tế phân tầng theo chuẩn BigCloneBench.
- **`download_bcb_subsets.py`**: Tải và trích xuất các tập dữ liệu BigCloneBench (Train 30k, Val 3k, Test 5k) từ kho CodeXGLUE.
- **`build_custom_suite_java.py`**: Khởi tạo tập kiểm thử phân tầng cho mã nguồn Java.
- **`build_custom_suite_negatives.py`**: Sinh các cặp mẫu âm tính (Không trùng lặp / Bẫy cú pháp) để kiểm thử ranh giới mô hình.

---

## 3. Trích xuất Cú pháp & Đồ thị (Feature Extraction)

- **`ast_canonicalizer.py`**: Chuẩn hóa cây cú pháp trừu tượng AST (bỏ comment, định danh biến chuẩn) cho ASTNN.
- **`fast_native_cpg.py`**: Bộ phân tích đồ thị thuộc tính chương trình (CPG) tốc độ cao phục vụ huấn luyện và suy luận GNN.
- **`build_bcb_features.py`**: Trích xuất hàng loạt đặc trưng đồ thị và luồng dữ liệu cho tập BigCloneBench.
- **`build_cpg_export.py`**: Tích hợp với công cụ Joern để xuất đồ thị CPG đa thành phần (AST + CFG + CDG + DDG).

---

## 4. Kiểm định & Đảm bảo Chất lượng (Audit & Verification)

- **`audit_bcb_dataset.py`**: Thống kê số lượng, độ dài và kiểm tra phân bố nhãn của tập BigCloneBench.
- **`check_bcb_leakage.py`**: Kiểm tra rò rỉ dữ liệu (Data Leakage) giữa các tập Train, Validation và Test.
- **`audit_dataset.py` & `check_leakage.py`**: Công cụ kiểm tra tính toàn vẹn và chồng lấn dữ liệu tổng quát.
- **`env_report.py`**: Kiểm tra cấu hình phần cứng, phiên bản CUDA, PyTorch và môi trường chạy thực nghiệm.

---

## 5. Sinh Notebooks Huấn luyện (Notebook Generators)

- **`generate_all_in_one_training_nb.py`**: Tự động sinh notebook huấn luyện trọn gói phục vụ chạy trên nền tảng điện toán đám mây (Kaggle T4 / P100 GPU).
- **`generate_m3_m4_training_nb.py`**: Sinh notebook chuyên biệt cho việc huấn luyện mô hình học cấu trúc (ASTNN và CPG-GNN).
- **`make_kaggle_freeze_backbone_nb.py`**: Sinh notebook huấn luyện phiên bản Đóng băng 12 tầng Transformer cho CodeBERT và GraphCodeBERT.
