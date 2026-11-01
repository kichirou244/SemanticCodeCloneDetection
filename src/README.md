# Thư mục Mã nguồn Cốt lõi (`src/`)

Chứa mã nguồn module hóa phục vụ huấn luyện, trích xuất đặc trưng và suy luận các mô hình phát hiện trùng lặp mã nguồn:

```text
src/
├── common/             # Cấu hình hệ thống (YAML loader) và seed kiểm soát ngẫu nhiên
├── core/               # Giao diện cơ bản (interfaces / abstract base classes)
├── data/               # Trình tải dữ liệu (BigCloneBench), chuẩn hóa mã nguồn, tạo cặp clone
├── experiments/        # Trình thực thi huấn luyện và đánh giá thực nghiệm (bcb_runner.py)
├── models/             # Kiến trúc 4 mô hình:
│   ├── transformer/    # Wrapper mô hình chuỗi (CodeBERT & GraphCodeBERT)
│   ├── astnn/          # Mô hình cây cú pháp ASTNN (RvNN + Bi-GRU)
│   ├── cpg_gnn/        # Mô hình đồ thị thuộc tính CPG-GNN (3-layer GAT)
└── representations/    # Trích xuất biểu diễn:
    ├── ast/            # Trích xuất cây cú pháp trừu tượng (Tree-sitter)
    └── cpg/            # Xây dựng và nạp đồ thị chương trình CPG
```
