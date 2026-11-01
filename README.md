# Semantic Code Clone Detection (Phát hiện Trùng lặp Mã nguồn Ngữ nghĩa)

Dự án nghiên cứu và đánh giá thực nghiệm các phương pháp Học sâu (Deep Learning) và Học đồ thị (Graph Neural Networks) trong bài toán **Phát hiện Trùng lặp Mã nguồn Ngữ nghĩa (Semantic Code Clone Detection)** trên hai ngôn ngữ lập trình **Java** và **C/C++**.

---

## 1. Giới thiệu Đề tài

Trùng lặp mã nguồn (Code Clone) là hiện tượng các đoạn mã có hành vi tương đồng hoặc giống hệt nhau. Trong đó, **Type-4 (Semantic Clones)** là nhóm trùng lặp phức tạp nhất khi hai đoạn mã có cú pháp, tên biến và giải thuật hoàn toàn khác biệt nhưng cùng thực thi chung một mục tiêu chức năng.

Đề tài tập trung triển khai, đối sánh và kiểm thử 4 phương pháp đại diện cho các cấp độ biểu diễn mã nguồn:

| Phương pháp | Cấp độ biểu diễn mã nguồn | Kiến trúc mô hình | Đặc tả |
| :--- | :--- | :--- | :--- |
| **M1: CodeBERT** | Dạng chuỗi (Sequence of Tokens) | Pre-trained RoBERTa | BPE Tokenizer, tối đa 256 tokens |
| **M2: GraphCodeBERT** | Chuỗi Token kết hợp Luồng dữ liệu (Tokens + DFG) | Graph-guided Masked Attention | Tích hợp các mối quan hệ biến (Data Flow Graph) |
| **M3: ASTNN** | Cây cú pháp trừu tượng (Abstract Syntax Tree) | RvNN + Bi-GRU | Cắt cây AST theo từng câu lệnh (Statement Trees) |
| **M4: CPG-GNN** | Đồ thị thuộc tính chương trình (Code Property Graph) | Graph Attention Network (GAT) | Tích hợp AST + CFG + CDG + DDG qua 3 tầng GAT |

---

## 2. Dữ liệu Thực nghiệm

Nghiên cứu sử dụng 2 bộ dữ liệu phục vụ huấn luyện và kiểm thử chéo ngôn ngữ:

1. **BigCloneBench (Java - Chuẩn CodeXGLUE):**
   - Trích xuất từ kho *IJaDataset-2.0*, gồm các hàm Java phân loại theo 43 nhóm chức năng thực tế.
   - Tập huấn luyện: **30.000** cặp (15.000 clone : 15.000 non-clone).
   - Tập kiểm định (Validation): **3.000** cặp (1.500 clone : 1.500 non-clone).
   - Tập kiểm thử độc lập (Test): **5.000** cặp (2.500 clone : 2.500 non-clone).

2. **CODER HUSC (C/C++ - Tập kiểm thử thực tế):**
   - Bộ dữ liệu gồm **400 cặp bài nộp sinh viên** thu thập từ hệ thống chấm tự động CODER HUSC trên các bài toán thuật toán.
   - Phân tầng nhãn chi tiết theo chuẩn BigCloneBench:
     - **Trùng lặp cú pháp (Type-1 / Type-2 / ST3):** 50 cặp
     - **Trùng lặp trung bình (MT3):** 30 cặp
     - **Trùng lặp ngữ nghĩa hoàn toàn (Type-4):** 120 cặp
     - **Không trùng lặp (Non-clone - Khác bài + Bẫy giải thuật):** 200 cặp

---

## 3. Kết quả Thực nghiệm

### 3.1. Kết quả trên Tập Test BigCloneBench (Java - 5.000 cặp)

| Mô hình / Phiên bản | Ngưỡng $\tau$ | Đúng Clone | Đúng Non-clone | F1-Score | Precision | Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 Chưa huấn luyện** | 0.01 | 2.500 / 2.500 | 0 / 2.500 | 66.67% | 50.00% | **100.00%** | 50.00% |
| **M1 Đóng băng 12 tầng** | 0.91 | 1.991 / 2.500 | 2.100 / 2.500 | 81.41% | 83.27% | 79.64% | 81.82% |
| **M1 Fine-tuning toàn bộ** | 0.88 | 2.234 / 2.500 | 2.426 / 2.500 | **92.93%** | 96.79% | 89.36% | **93.20%** |
| **M2 Chưa huấn luyện** | 0.93 | 2.089 / 2.500 | 902 / 2.500 | 67.53% | 56.66% | 83.56% | 59.82% |
| **M2 Đóng băng 12 tầng** | 0.10 | 2.189 / 2.500 | 2.171 / 2.500 | 87.25% | 86.93% | 87.56% | 87.20% |
| **M2 Fine-tuning toàn bộ** | 0.94 | 2.197 / 2.500 | 2.458 / 2.500 | 92.72% | **98.12%** | 87.88% | 93.10% |
| **M3 Chưa huấn luyện** | 0.87 | 2.500 / 2.500 | 7 / 2.500 | 66.73% | 50.07% | **100.00%** | 50.14% |
| **M3 Fine-tuning toàn bộ** | 0.52 | 2.222 / 2.500 | 449 / 2.500 | 65.61% | 52.00% | 88.88% | 53.42% |
| **M4 Chưa huấn luyện** | 0.01 | 2.500 / 2.500 | 0 / 2.500 | 66.67% | 50.00% | **100.00%** | 50.00% |
| **M4 Fine-tuning toàn bộ** | 0.70 | 2.500 / 2.500 | 0 / 2.500 | 66.67% | 50.00% | **100.00%** | 50.00% |

### 3.2. Đánh giá Zero-shot Cross-Language trên CODER HUSC (C/C++ - 400 cặp)

Mô hình được huấn luyện trên mã nguồn Java và kiểm thử trực tiếp trên mã nguồn C/C++:

| Mô hình / Phiên bản | Ngưỡng $\tau$ | Type-1/2/ST3 (50) | MT3 (30) | Type-4 (120) | Non-clone (200) | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1 Đóng băng 12 tầng** | 0.01 | 50 / 50 | 30 / 30 | 119 / 120 | 0 / 200 | 66.44% | 49.75% |
| **M1 Fine-tuning toàn bộ** | 0.97 | 47 / 50 | 29 / 30 | 115 / 120 | 28 / 200 | 67.85% | 54.75% |
| **M2 Đóng băng 12 tầng** | 0.07 | 43 / 50 | 27 / 30 | 106 / 120 | 61 / 200 | 68.35% | 59.25% |
| **M2 Fine-tuning toàn bộ** | 0.78 | 46 / 50 | 25 / 30 | 111 / 120 | 50 / 200 | 68.42% | 58.00% |
| **M3 Fine-tuning toàn bộ** | 0.36 | 50 / 50 | 29 / 30 | 104 / 120 | 65 / 200 | **70.66%** | **62.00%** |
| **M4 Fine-tuning toàn bộ** | 0.01 | 50 / 50 | 30 / 30 | 120 / 120 | 0 / 200 | 66.67% | 50.00% |

---

## 4. Cấu trúc Dự án

```text
SemanticCodeCloneDetection/
├── configs/                     # Cấu hình huấn luyện mô hình (YAML)
├── data/                        # Dữ liệu kiểm thử
│   └── processed/
│       └── coder_benchmark/     # Bộ benchmark 400 cặp C/C++ CODER HUSC
├── notebooks/                   # Jupyter notebooks huấn luyện trên Kaggle GPU
│   └── README.md                # Mô tả chi tiết từng notebook
├── reports/                     # Báo cáo học thuật (LaTeX source và file PDF)
│   ├── [31 - 10] - Báo cáo.tex  # Mã nguồn LaTeX
│   └── [31 - 10] - Báo cáo.pdf  # Bản PDF xuất bản
├── results/                     # Kết quả đánh giá đã ghi nhận (JSON)
│   ├── bcb_test_5k_results.json # Kết quả trên 5.000 cặp BigCloneBench Test
│   └── coder_benchmark_results.json # Kết quả trên 400 cặp CODER HUSC
├── scripts/                     # Kịch bản dòng lệnh thực thi pipeline
│   ├── README.md                # Hướng dẫn chi tiết từng script
│   ├── evaluate_bcb_5000_test.py    # Đánh giá 10 biến thể trên 5.000 cặp Java
│   ├── evaluate_coder_benchmark.py  # Đánh giá trên 400 cặp C/C++ thực tế
│   ├── crawl_coder_husc.py          # Thu thập dữ liệu bài nộp CODER HUSC
│   └── prepare_coder_benchmark.py   # Xây dựng bộ benchmark C/C++
├── src/                         # Mã nguồn cốt lõi (Core Modules)
│   ├── README.md                # Sơ đồ các module trong src/
│   ├── common/                  # Quản lý cấu hình và seed ngẫu nhiên
│   ├── core/                    # Giao diện cơ bản (interfaces)
│   ├── data/                    # Loader BigCloneBench và trình sinh cặp
│   ├── experiments/             # Trình điều phối huấn luyện (bcb_runner.py)
│   ├── models/                  # Định nghĩa 4 kiến trúc (M1, M2, M3, M4)
│   └── representations/         # Trích xuất biểu diễn AST và CPG
├── tests/                       # Unit tests kiểm thử tính đúng đắn
├── requirements.txt             # Danh sách thư viện phụ thuộc
└── README.md                    # Tài liệu tổng quan đề tài
```

---

## 5. Hướng dẫn Chạy Thực nghiệm

### 5.1. Khởi tạo Môi trường

Khuyến nghị sử dụng Python 3.10+:

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 5.2. Chạy Tái hiện Đánh giá

1. **Đánh giá trên tập kiểm thử BigCloneBench (Java - 5.000 cặp):**
   ```bash
   python scripts/evaluate_bcb_5000_test.py
   ```
   Kết quả xuất ra tại `results/bcb_test_5k_results.json`.

2. **Đánh giá trên tập kiểm thử CODER HUSC (C/C++ - 400 cặp):**
   ```bash
   python scripts/evaluate_coder_benchmark.py
   ```
   Kết quả phân tầng chi tiết xuất ra tại `results/coder_benchmark_results.json`.
