# Task log — SemanticCodeCloneDetection

## [COMPLETE] Báo cáo `[09 - 10]`
- Tạo `reports/[09 - 10] - Báo cáo.tex` từ `[06-10]-Report.tex`, xoá đoạn nhận xét sau bảng huấn luyện.
- Gộp mục 1+2 thành mục 1 `Kết quả tuần này` (hoàn thành train 5 PP + demo trên tập test ở mục 6).
- Sửa chữ không dấu: `(khong clone)` → `(không clone)` trong `workspace/demo_neg5.tex` (+ file txt đi kèm), `model` → `mô hình` trong báo cáo.
- Build PDF 7 trang xong.

## [COMPLETE] Tối ưu train V2 (vừa quota Kaggle 10h)
- M5 Hybrid (`configs/exp006_hybrid.yaml` + `src/experiments/exp006_hybrid.py`):
  LR head 2e-5 → 1e-5, thêm `encoder_learning_rate: 5e-6` (LR phân biệt encoder/head),
  `epochs` 5 → 3. Lý do: E0 đỉnh 0,914 rồi train loss sập 0,054 → 0,003 (overfit).
- M4 CPG-GNN (`configs/exp005_cpg_gnn.yaml`): `patience` 2 → 4, giữ `epochs: 15`
  (bản báo cáo dừng ở epoch 5, val F1 còn tăng 0,780 → 0,819).
- Test: 33 passed (`test_exp006_hybrid` + `test_configs` + `test_cpg_gnn`).
- Notebook Kaggle:
  - `exp005_cpg_gnn_kaggle.ipynb` cell 9 = FULL TRAIN từ đầu, budget 3,5h.
  - `exp006_hybrid_kaggle.ipynb` cell 10 = train MỚI hoàn toàn (không resume), budget 7h.

## [TO DO] Chạy trên Kaggle (user)
1. Import 2 notebook, `git pull` ở cell 2.
2. Chạy CPG trước (~3h), rồi Hybrid (~7-8h).
3. Backup `results/exp006_hybrid_poj104/best_checkpoint.pt` cũ nếu cần giữ số báo cáo 09-10
   (Hybrid train mới ghi đè thư mục kết quả).

## [PLAN] Sau khi có số mới
- Cập nhật bảng val F1 + test trong `reports/[09 - 10] - Báo cáo.tex`, build lại PDF.
- So sánh Hybrid V2 vs V1 (hết overfit E0 chưa), CPG epoch mới có tăng tiếp không.

## [BACKLOG] Tạm hoãn (vượt quota)
- Thử đầu vào 512 token, nới giới hạn node CPG-GNN (attention O(n²), một mình
  retrain CodeBERT đã nuốt hết ~10h).
- Two-stage Hybrid: freeze encoder 1 epoch đầu rồi unfreeze (nếu V2 vẫn overfit).
