# Thư mục Kịch bản Thực thi (`scripts/`)

Tập trung các kịch bản thực thi trực tiếp từ dòng lệnh cho toàn bộ pipeline:

```text
scripts/
├── evaluate_bcb_5000_test.py    # Đánh giá 10 biến thể trên 5.000 cặp Test BigCloneBench (Java)
├── evaluate_coder_benchmark.py  # Đánh giá khả năng chuyển dịch trên 400 cặp CODER HUSC (C/C++)
├── prepare_coder_benchmark.py   # Chuẩn hóa & tạo tập test phân tầng CODER từ mã nộp sinh viên
├── crawl_coder_husc.py          # Thu thập tự động các bài nộp từ hệ sinh thái CODER HUSC
├── download_bcb_subsets.py      # Tải các phân đoạn BigCloneBench từ CodeXGLUE
├── audit_bcb_dataset.py         # Kiểm định phân bố nhãn và thống kê tập dữ liệu
├── check_bcb_leakage.py         # Kiểm tra tránh rò rỉ dữ liệu (leakage) giữa train/val/test
├── build_bcb_features.py        # Trích xuất đặc trưng AST & DFG cho BigCloneBench
├── fast_native_cpg.py           # Parser đồ thị CPG tốc độ cao
├── build_cpg_export.py          # Xuất đồ thị CPG bằng công cụ Joern
├── make_cpg_shards.py           # Phân mảnh lưu trữ CPG để tối ưu bộ nhớ
├── generate_bcb_notebooks.py    # Sinh các notebook huấn luyện độc lập cho Kaggle
└── env_report.py                # Báo cáo thông tin môi trường GPU, CUDA và thư viện
```
