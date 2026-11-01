# Thư mục Kịch bản Thực thi (`scripts/`)

Các kịch bản thực thi trực tiếp từ dòng lệnh phục vụ toàn bộ pipeline của đề tài:

```text
scripts/
├── evaluate_bcb_5000_test.py    # Đánh giá 10 biến thể trên 5.000 cặp Test BigCloneBench (Java)
├── evaluate_coder_benchmark.py  # Đánh giá trên 400 cặp C/C++ thực tế CODER HUSC
├── prepare_coder_benchmark.py   # Chuẩn hóa & tạo tập test phân tầng CODER HUSC
├── crawl_coder_husc.py          # Thu thập dữ liệu bài nộp sinh viên từ CODER HUSC
├── download_bcb_subsets.py      # Tải dữ liệu BigCloneBench từ CodeXGLUE
├── build_bcb_features.py        # Trích xuất đặc trưng AST & DFG cho BigCloneBench
├── fast_native_cpg.py           # Parser đồ thị CPG đa thành phần tốc độ cao
├── build_cpg_export.py          # Xuất đồ thị CPG chuẩn hóa bằng Joern
└── env_report.py                # Báo cáo thông tin phần cứng GPU, CUDA và môi trường
```
