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
- **Tiêu chí hoàn thành (DoD):** Unit test `tests/test_ast_extract_java.py` đạt 100% pass; Đã chạy hoàn tất trên Kaggle CPU (`kichirou/semanticcodeclonedetection` Version 39), biên dịch và lưu thành công `bcb_ast_cache.npz` (1.2 MB, 9.126 hàm Java duy nhất từ toàn bộ 1.731.860 cặp BigCloneBench, tốc độ 1.076 hàm/s).

### [BCB-04] Chuyển đổi pipeline sinh CPG Java bằng Joern cho CPG-GNN (M4) & Hybrid (M5)
- **Trạng thái:** Done
- **Mục tiêu:** Tạo pipeline sinh Code Property Graph (CPG) cho các hàm Java bằng Joern CLI.
- **Mô tả chi tiết:**
  - Joern hỗ trợ Java gốc rất mạnh thông qua frontend `javasrc2cpg` (vượt trội hơn C/C++ do không bị lỗi thiếu include header).
  - Cập nhật script `src/representations/cpg/ingest.py` để xuất đồ thị CPG đa cạnh (AST edge, Control Flow Graph - CFG, Data Dependency Graph - DDG/CDG) cho các hàm `.java`.
  - Chuẩn hóa schema đồ thị tương thích với PyTorch Geometric: định nghĩa feature ma trận node (loại node, code token) và tensor ma trận kề cạnh theo từng loại quan hệ.
- **Tiêu chí hoàn thành (DoD):** Chạy thử nghiệm trích xuất 200 đồ thị Java mẫu; nạp thành công vào lớp `Data` của PyTorch Geometric mà không gặp xung đột kiểu dữ liệu. Đã xuất thành công toàn bộ 9.122 đồ thị CPG bằng Joern `javasrc2cpg` trên Kaggle CPU (`kichirou/semanticcodeclonedetection` Version 41) và đóng gói vào `bcb_cpg_export.tar.gz` (21.4 MB, 100% CPU, 0% GPU quota).

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
- **Trạng thái:** Done
- **Mục tiêu:** Đánh giá định lượng toàn diện các mô hình trên tập test Java BigCloneBench và phân tích chi tiết theo loại clone.
- **Mô tả chi tiết:**
  - Tính toán các chỉ số cốt lõi: Precision, Recall, F1, Accuracy tại ngưỡng tối ưu chốt từ tập Validation.
  - Đánh giá trên bộ dữ liệu kiểm thử thực tế 720 cặp (tỉ lệ 1:5) gồm Type-1, Type-2, Type-3, Type-4 và các cạm bẫy Non-clone khó.
  - Phân tích chi tiết số lượng đoán đúng của Clone (`TP / 120`) và Non-clone (`TN / 600`).
- **Tiêu chí hoàn thành (DoD):** Sinh file tổng hợp `workspace/custom_suite_java_scores.json` và bảng so sánh chi tiết.

### [BCB-10] Chuyển đổi Custom Test Suite Java (720 cặp, tỉ lệ 1:5) & Phân tích cạm bẫy
- **Trạng thái:** Done
- **Mục tiêu:** Cung cấp bộ test thực tế nhỏ gọn có thể giải thích trực quan và kiểm tra độ vững cáp trước cạm bẫy cú pháp.
- **Mô tả chi tiết:**
  - Xây dựng bộ test 720 cặp Java gồm 120 cặp clone (Type-1 đến Type-4) và 600 cặp non-clone (trong đó có 75 cặp non-clone cạm bẫy khung sườn).
  - So sánh đối chứng trực tiếp với Stanford MOSS: MOSS chỉ bắt được 11,1% Type-4 trong khi GraphCodeBERT bắt được 88,9%.
- **Tiêu chí hoàn thành (DoD):** Hoàn thành script chấm tự động `scripts/score_custom_suite_java.py` và lưu trữ tại `data/custom_suite_java/`.

### [BCB-11] Chốt số liệu & xuất Báo cáo tuần (31-10-2026)
- **Trạng thái:** Done
- **Mục tiêu:** Hoàn thiện hồ sơ tài liệu khoa học và đồng bộ tiến độ dự án.
- **Mô tả chi tiết:**
  - Tổng hợp toàn bộ thông số huấn luyện (Bảng 2) và kết quả kiểm thử đối chứng các biến thể: Chưa huấn luyện, Đóng băng 12 tầng, Fine-tuning toàn bộ (Bảng 3).
  - Biên dịch báo cáo chuẩn XeLaTeX 2 trang A4 sạch đẹp tại `reports/[31 - 10] - Báo cáo.pdf`.
- **Tiêu chí hoàn thành (DoD):** File PDF báo cáo biên dịch sạch (0 error, đúng 2 trang A4, đã đồng bộ lên GitHub `main`).

---

## 5. Kế hoạch tuần tiếp theo (03-11 → 07-11-2026): Tối ưu hóa mô hình & Xử lý mất cân bằng thực tế

### [OPT-01] Áp dụng LoRA / PEFT cho GraphCodeBERT (M2) & CodeBERT (M1)
- **Trạng thái:** In planning
- **Mục tiêu:** Giải quyết bài toán đánh đổi giữa Recall và Precision, kết hợp ưu điểm của Đóng băng (không quên tri thức gốc, TN cao) và Fine-tuning (bắt Type-4 nhạy, Recall cao).
- **Mô tả chi tiết:**
  - Tích hợp thư viện `peft` (Hugging Face) vào `src/models/transformer_pair.py`.
  - Giữ cố định 100% trọng số Transformer backbone, chỉ huấn luyện các ma trận thích ứng rank thấp (LoRA với $r=8$ hoặc $r=16$, $\alpha=32$) tại các module Attention (`q_proj`, `v_proj`).
  - Đánh giá khả năng chặn False Positives trên 75 ca cạm bẫy non-clone mà vẫn duy trì Recall $\ge 90\%$.
- **Tiêu chí hoàn thành (DoD):** Script huấn luyện LoRA chạy trên Kaggle GPU; checkpoint weights siêu nhẹ (< 20MB); F1 trên tập kiểm thử 1:5 cải thiện vượt ngưỡng 75%.

### [OPT-02] Khai thác Hard Negative Mining & Hàm mất mát Focal Loss / Contrastive Loss
- **Trạng thái:** In planning
- **Mục tiêu:** Giảm thiểu triệt để số lượng báo động giả (False Positives) trên các cặp mã có bề mặt code tương tự nhưng khác thuật toán.
- **Mô tả chi tiết:**
  - Thu thập và khai thác tập mẫu âm khó (Hard Negatives): các cặp mã khác bài nhưng dùng chung import, khung hàm hoặc cùng giải thuật đệ quy nhưng trả lời bài toán khác nhau.
  - Cập nhật hàm mất mát trong `src/experiments/bcb_runner.py`: thử nghiệm `FocalLoss(gamma=2.0, alpha=0.25)` và `SupervisedContrastiveLoss` để tăng mức phạt đối với các ca cạm bẫy đoán sai.
- **Tiêu chí hoàn thành (DoD):** Số ca FP của M1/M2 trên tập non-clone giảm tối thiểu 50%.

### [OPT-03] Tối ưu hóa cấu trúc cho M3 (ASTNN) & M4 (CPG-GNN)
- **Trạng thái:** In planning
- **Mục tiêu:** Tháo gỡ nút thắt cổ chai về độ sâu cây cú pháp và kích thước đồ thị CPG.
- **Mô tả chi tiết:**
  - **M3 (ASTNN):** Áp dụng AST Pruning (cắt tỉa node rác, loại bỏ khai báo kiểu rườm rà) và AST Canonicalization (chuẩn hóa `while` $\leftrightarrow$ `for`, chuẩn hóa tên biến tạm).
  - **M4 (CPG-GNN):** Áp dụng Program Slicing để trích xuất lát cắt đồ thị theo biến trọng yếu; thử nghiệm Heterogeneous GNN (RGCN) phân biệt 3 loại cạnh: AST, CFG, DDG.
- **Tiêu chí hoàn thành (DoD):** F1 của M3 vượt 45%; M4 vượt 60% trên tập dữ liệu kiểm thử.

### [OPT-04] Đánh giá trên tập dữ liệu Coder / Cross-lingual khi tiếp nhận dữ liệu mới
- **Trạng thái:** In planning
- **Mục tiêu:** Kiểm chứng khả năng thích ứng miền và chuyển giao tri thức mã nguồn đa ngôn ngữ trên dữ liệu thực tế mới.
- **Mô tả chi tiết:**
  - Ngay khi tiếp nhận tập dữ liệu Coder: nạp dữ liệu vào pipeline chuẩn hóa `src/data/loaders/`.
  - Chạy suy luận tự động qua các checkpoint tốt nhất của M1, M2 (Full, Freeze, LoRA).
  - Xuất bảng kết quả đối chứng đa chiều so sánh giữa Java, C/C++ và Coder.
- **Tiêu chí hoàn thành (DoD):** Báo cáo so sánh hiệu năng trên tập dữ liệu mới hoàn thành trong vòng 24h sau khi nhận dữ liệu.
