# Danh mục Kịch bản Thực thi (Scripts Reference)

Thư mục `scripts/` chứa toàn bộ các kịch bản thực thi phục vụ thu thập dữ liệu, xử lý đặc trưng, sinh notebook huấn luyện và đánh giá thực nghiệm cho bài toán Phát hiện Trùng lặp Mã nguồn (BigCloneBench & CODER HUSC).

---

## 1. Đánh giá & Benchmark (Evaluation & Metrics)

- **`evaluate_bcb_5000_test.py`**: Đánh giá 10 biến thể mô hình (M1, M2, M3, M4 với Chưa huấn luyện, Đóng băng 12 tầng, Fine-tuning) trên đúng **5.000 cặp Test độc lập của BigCloneBench**. Kết quả lưu tại `results/bcb_test_5k_results.json`.
- **`evaluate_coder_benchmark.py`**: Đánh giá khả năng chuyển dịch chéo ngôn ngữ (Cross-language Zero-shot) trên **400 cặp C/C++ từ CODER HUSC**, phân loại chi tiết theo: Type-1/2/ST3, MT3, Type-4, Non-clone. Kết quả lưu tại `results/coder_benchmark_results.json`.
- **`score_custom_suite_java.py`**: Đánh giá các mô hình trên tập kiểm thử phân tầng Java.
- **`score_custom_suite_cpp.py`**: Đánh giá các mô hình trên tập kiểm thử C/C++.
- **`eval_moss_service.py`**: Đối sánh kết quả với hệ thống phát hiện gian lận MOSS (Stanford University).

---

## 2. Thu thập & Xây dựng Dữ liệu (Data Preparation)

- **`crawl_coder_husc.py`**: Thu thập dữ liệu bài nộp sinh viên C/C++ từ hệ thống chấm tự động CODER HUSC.
- **`prepare_coder_benchmark.py`**: Chuẩn hóa mã nguồn và tạo bộ dữ liệu 400 cặp kiểm thử thực tế phân tầng theo chuẩn BigCloneBench.
- **`download_bcb_subsets.py`**: Tải và trích xuất các tập dữ liệu BigCloneBench (Train 30k, Val 3k, Test 5k) từ kho CodeXGLUE.
- **`build_custom_suite_java.py`**: Khởi tạo tập kiểm thử phân tầng cho mã nguồn Java.

---

## 3. Trích xuất Cú pháp & Đồ thị (Feature Extraction)

- **`ast_canonicalizer.py`**: Chuẩn hóa cây cú pháp trừu tượng AST (bỏ chú thích, chuẩn hóa định danh) cho ASTNN.
- **`fast_native_cpg.py`**: Phân tích đồ thị thuộc tính chương trình (CPG) tốc độ cao phục vụ huấn luyện và suy luận GNN.
- **`build_bcb_features.py`**: Trích xuất đặc trưng AST và luồng dữ liệu cho tập BigCloneBench.
- **`build_cpg_export.py`**: Tích hợp công cụ Joern để xuất đồ thị CPG đa thành phần (AST + CFG + CDG + DDG).
- **`make_cpg_shards.py`**: Chia nhỏ các đồ thị CPG thành từng phân đoạn để xử lý tối ưu bộ nhớ.

---

## 4. Kiểm định & Đảm bảo Chất lượng (Audit & Verification)

- **`audit_bcb_dataset.py`**: Thống kê số lượng, độ dài và phân bố nhãn của tập BigCloneBench.
- **`check_bcb_leakage.py`**: Kiểm tra rò rỉ dữ liệu (Data Leakage) giữa các tập Train, Validation và Test.
- **`env_report.py`**: Kiểm tra thông tin phần cứng, phiên bản CUDA, PyTorch và môi trường thực nghiệm.

---

## 5. Sinh Notebooks Huấn luyện (Notebook Generators)

- **`generate_all_in_one_training_nb.py`**: Sinh notebook huấn luyện trọn gói phục vụ chạy trên Kaggle GPU (T4 / P100).
- **`generate_bcb_notebooks.py`**: Sinh notebook chạy từng mô hình riêng biệt trên BigCloneBench.
- **`generate_bcb_feature_extraction_nb.py`**: Sinh notebook trích xuất đặc trưng BigCloneBench trên Kaggle.
- **`generate_m3_m4_training_nb.py`**: Sinh notebook chuyên biệt cho ASTNN và CPG-GNN.
- **`make_kaggle_freeze_backbone_nb.py`**: Sinh notebook huấn luyện phiên bản Đóng băng 12 tầng Transformer.
