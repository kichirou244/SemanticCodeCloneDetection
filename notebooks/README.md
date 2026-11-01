# Thư mục Sổ tay Thí nghiệm (`notebooks/`)

Tập hợp các notebook định dạng Jupyter phục vụ huấn luyện và kiểm thử phân tán trên hạ tầng Kaggle GPU (T4 / P100):

```text
notebooks/
├── bcb_all_models_kaggle.ipynb                # Huấn luyện toàn diện 4 mô hình trong một phiên
├── bcb_exp-bcb-002_codebert_kaggle.ipynb      # Huấn luyện riêng mô hình M1: CodeBERT
├── bcb_exp-bcb-003_graphcodebert_kaggle.ipynb # Huấn luyện riêng mô hình M2: GraphCodeBERT
├── bcb_exp-bcb-004_astnn_kaggle.ipynb         # Huấn luyện riêng mô hình M3: ASTNN
├── bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb       # Huấn luyện riêng mô hình M4: CPG-GNN
├── bcb_feature_extraction_kaggle.ipynb        # Trích xuất trước đặc trưng AST và CPG
└── bcb_m3_m4_kaggle.ipynb                     # Huấn luyện cặp mô hình học cấu trúc (M3 + M4)
```

> **Ghi chú:** Các notebook được thiết kế khép kín (self-contained), tự động tải dependencies, checkpointing và lưu metric đánh giá ra các tệp `.json` kết quả.
