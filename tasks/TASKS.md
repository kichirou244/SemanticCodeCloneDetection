# Task log — SemanticCodeCloneDetection

## [COMPLETE] Dừng train → pipeline cuối: đánh giá + chọn + suy luận (17-10)
- Quyết định **dừng huấn luyện** (các bản 512 đã chạm trần thực dụng, chênh còn lại
  nằm trong nhiễu) và chuyển sang đánh giá để chốt phương pháp → DEC-030.
- Mới: `configs/final_pipeline.yaml`, `src/evaluation/leaderboard.py`,
  `scripts/final_pipeline.py` (`select`/`detect`), `tests/test_final_pipeline.py`
  (14 test, không cần GPU), `reports/014_final_selection.md`, artifact
  `results/final_pipeline/`.
- `select` xếp hạng 10 run theo **val F1** (ngưỡng đóng băng), so winner vs á quân
  bằng **McNemar + paired bootstrap tái dùng `uncertainty.py`** trên validation.
- Kết quả: **M5-512 thắng theo val F1** (0,9379) nhưng hơn M2-512 chỉ +0,0043;
  McNemar p=0,0814 (không ý nghĩa) vs bootstrap CI95 [+0,0004,+0,0083] (chạm 0) ⇒
  `mixed`; ngưỡng 0,010 rất mỏng; **M2-512 hơn ở test F1 0,9550 / AUC 0,9912** ⇒
  **pipeline cuối dùng M2-512 cho `detect`** (`inference.default_model: M2_512`;
  M5-512 gọi khi cần bằng `--model M5_512`).
- Kế hoạch train tuần 18–23/10 (Hybrid V3, CPG nới node, Hybrid-512) **tạm gác** —
  chỉ mở lại nếu cần bằng chứng mạnh hơn cho lựa chọn.
- Chạy lại: `python scripts/final_pipeline.py select` ·
  `python scripts/final_pipeline.py detect --a a.cpp --b b.cpp`

## [COMPLETE] Demo + báo cáo chuyển sang bản 512 cho M1/M2/M5 (17-10) — DEC-031
- `configs/demo_web.yaml`: M1/M2/M5 trỏ `results/exp00{2,3,6}_*_512` (`max_length: 512`,
  `best_checkpoint.pt`; ngưỡng vẫn đọc từ `threshold.json` của chính run). M3/M4/B1 giữ
  nguyên vì không có biến thể 512 (AST / node CPG / winnowing đều không phụ thuộc
  `max_length`).
- Chấm lại bộ test: `python scripts/score_custom_suite.py --models M1 M2 M5 --merge`
  (giữ B1/M3/M4) rồi `python scripts/evidence_custom_suite.py --reuse-moss --pairs
  P1_combo P3_nonclone_sum P1_nonclone_fact` (giữ lại artifact MOSS service thật).
- Kết quả trên 26 cặp: lỗi M2 3→**2**, M5 4→**3** (sửa 2 dương giả ở `P2_nonclone_digitsum`),
  **không cặp nào đổi từ đúng sang sai**; §4 báo cáo `[17 - 10]` cập nhật số mới.
- Guard: `tests/test_demo_registry.py::test_transformer_models_use_512_runs`.

## [QUY UOC] Bo test mac dinh: `data/custom_suite` (tu 16-10)
- Moi lan test/demo/bao cao sau nay deu dung `data/custom_suite`
  (26 cap: 16 clone + 10 non-clone), cham bang `scripts/score_custom_suite.py`,
  sinh hinh bang `scripts/evidence_custom_suite.py`. Khong them comment vao file
  test (da xoa sach 16-10). `data/custom_hard/` giu lai lam tham khao.
- Ket qua full 16-10 (khong comment): 16/16 clone ca 5 model neural dung;
  non-clone kho: P1_fact + P4_square ca 5 model sai, P2_digitsum chi M1 dung,
  P3_sum M1/M2/M5 dung; cross X: M1/M2 6/6, M5 4/6, M3/M4 0/6. B1-service 0 diem
  moi cap (mu Type-3/4, da biet tu EXP-014).
- Bao cao [17-10]: muc 6 co 3 cap (6.1 clone kho P1_combo, 6.2 non-clone kho
  P3_nonclone_sum, 6.3 bay skeleton P1_nonclone_fact ca 5 model sai).
  Khong ton tai ca false-non-clone nhat tri (16/16 clone deu bi bat).

## [QUY UOC] Cập nhật trạng thái task trên Notion (từ 18-10)
- Trước khi bắt đầu 1 việc: mở task trong Notion, đổi Status -> **In progress**.
- Làm xong: đổi Status -> **Done** (ghi kết quả vào Description nếu cần).
- Xem task: `node tasks/notion.js --tasks`
- Đổi status: `node tasks/notion.js --status "<tên task>" "<trạng thái>"`
  (trạng thái: Not started / In progress / On Hold / To Review / Done / Cancelled)
- Sync kế hoạch từ `tasks/tasks.json` lên Notion: `node tasks/notion.js`
  (khớp theo tên: có thì cập nhật, chưa có thì tạo mới, không tạo trùng).

## [TO DO] Kế hoạch tuần 18-10 (CN) → 23-10 (T6), T7 nộp báo cáo
Mục tiêu: (1) Hybrid hết overfit đỉnh-ngay-E0 bằng freeze encoder rồi unfreeze;
(2) CPG-GNN nới giới hạn node để bớt cắt hàm dài; (3) thử cấu hình khác cho Hybrid-512.
Gom theo từng mạch việc (config → chạy → đánh giá), không tách vụn.

### 1. Hybrid V3 (freeze→unfreeze) — 18-10 → 20-10
- 18-10: sửa `src/experiments/exp006_hybrid.py` + `configs/exp006_hybrid.yaml` thêm
  `freeze_encoder_epochs` (1–2); viết/sửa test, `pytest` xanh.
- 19-10: chạy notebook Kaggle (~7–8h), theo dõi val F1 từng epoch (kỳ vọng ≥ V1 0,914,
  hết đỉnh-ngay-E0).
- 20-10: so val/test F1 với V1 (0,913/thr 0,370) và V2, quyết định giữ V1 hay chuyển V3.

### 2. CPG-GNN nới node (max_nodes 1500→2000) — 18-10 → 21-10
- 18-10: soạn config `exp005_cpg_gnn` bản nới node.
- 20-10: chạy M4 trên Kaggle.
- 21-10: đánh giá val F1 có vượt 0,842 không + tỉ lệ cắt node (`truncation.json`).

### 3. Hybrid-512 cấu hình mới — 21-10 → 22-10
- 21-10: chạy cấu hình mới (freeze encoder / epochs / LR khác).
- 22-10: đánh giá, so M5-512 (test F1 0,952, thr 0,010 rất mỏng) + xem cả verdict.

### 4. Truncation M1/M2/M5-512 — 22-10
- Đo lại `truncation.json` bản 512 so với 256.

### 5. Chốt số + báo cáo tuần (T7 nộp) — 23-10
- Chấm lại `data/custom_suite` cho model mới bằng `scripts/score_custom_suite.py`.
- Cập nhật bảng train/test + mục demo trong báo cáo tuần, build PDF để T7 nộp.

## [COMPLETE] Bổ sung MOSS + ví dụ khó theo review báo cáo [10-10]
- Đào từ `test_scores.jsonl` (ngưỡng threshold.json): 11 cặp FN cả 5 mô hình,
  65 cặp FP cả 5 mô hình. Chọn `test-pos-30-284` (GT=clone, 0,073/0,022/0,407/
  0,292/0,002) và `test-neg-218` (GT=non-clone, 0,998/0,999/0,734/0,828/0,999).
- Tự tạo `data/custom_hard/`: `base_sum7.cpp` + 3 biến thể clone (for→while,
  cin→scanf, combo rename+do-while+dead code) + 1 non-clone khó (cùng skeleton,
  `cnt+=1` thay `sum+=i*i`). Chấm local M1/M2/M3 + MOSS-local (k=20,w=5):
  3 cặp clone đều đúng; cặp non-clone khó cả 4 đều sai (1,0/1,0/0,978/0,795).
- MOSS-local: FN POJ 0,186 (đúng — thắng cả 5 model), FP POJ 0,033 (sai cùng).
  M4/M5 cặp tự tạo chưa chấm local được (thiếu Java/Joern).
- Báo cáo: thêm hàng MOSS vào bảng test + mục 6.3–6.6, build PDF 8 trang
  (`[09-10]` .tex/.pdf, copy sang `[10-10]` .pdf).

## [COMPLETE] Review lần 2: gọn 6.3+, MOSS highlight, bộ test tái dùng
- Bỏ hàng MOSS khỏi bảng test (theo review). Minh họa MOSS = highlight k-gram
  chung render bằng pipeline evidence có sẵn (`svc.evidence_for("B1")` local +
  `_latex_minipage` trong `scripts/scan_demo_zip.py`) → `workspace/demo_moss_custom.tex`
  (2 cặp C6/N1), không gọi service Stanford.
- Bộ test mở rộng 8 cặp: `data/custom_hard/pairs.jsonl` (manifest) + 4 file mới
  (dowhile, reorder_dead, while_scanf, max_track) + `scripts/score_custom_hard.py`
  (chạy lại bất cứ lúc nào) → `workspace/custom_hard_scores.json`.
  Kết quả: 6 clone đúng hết (M3 0,777–0,963), 2 non-clone khó sai hết.
- Báo cáo viết lại 6.3 (bảng 8 cặp + cách tái dùng), 6.4 (bảng 2 ca POJ cả-5-sai),
  6.5 (MOSS highlight). Build PDF 8 trang, copy sang `[10-10]` .pdf.

## [COMPLETE] MOSS server thật cho bộ test khó (user cấp id)
- Sửa bug do đặt id: `_ENV_USER_ID` phải là tên biến môi trường, id vào
  `_DEFAULT_USER_ID` (env `MOSS_USER_ID` được ưu tiên nếu có).
- Nộp gộp 13 file / 1 lần lên `moss.stanford.edu`
  (report `http://moss.stanford.edu/results/9/6309937295477`):
  10 cặp chỉ 3 bị gắn cờ — C2 (0,61 đúng), N1 (0,75 sai), N2 (0,55 sai);
  mọi biến thể đổi vòng lặp + cả 2 cặp POJ đều 0 điểm.
- Báo cáo chuyển cột MOSS sang điểm server + highlight từ trang match Stanford
  (`workspace/demo_moss_custom.tex` viết lại, `moss_service_scores.json`).
  Kết luận: MOSS thật mù Type-3/4 (5/6 clone tự tạo + FN POJ đều 0 điểm).
- Minh họa 6.1/6.2 thêm MOSS server (`workspace/demo_moss_61.tex`,
  `demo_moss_62.tex` từ trang match Stanford: 6.1 → 0,980 đúng, 6.2 → 0,000 đúng).
  Sửa lỗi build LaTeX có từ trước (babel `"` trong code mẫu, `_` trong tiêu đề):
  build còn 0 error. PDF 10 trang, copy sang `[10-10]` .pdf.
- Theo yêu cầu: xóa mục 6.3–6.5 (bộ test tự tạo + 2 ca POJ + MOSS highlight custom).
  Báo cáo còn 6.1/6.2 (5 model + MOSS server), 8 trang, build 0 error.
  Bộ test `data/custom_hard/` + `scripts/score_custom_hard.py` vẫn giữ trong repo
  để dùng lại sau này.

## [COMPLETE] Hậu kỳ highlight saliency (Top-K + stop-words + merge vệt)
- Đo trên saliency thật M1 (2 cặp × 2 phía): baseline (>mean) cov 23–30%,
  punct 4–11%, 8–15 vệt; Top-15% → cov 15%, punct ~0%, 6–11 vệt.
  Gom theo dòng tuy ít vệt (3–6) nhưng cov phình 46–55% (lôi cả dấu ngoặc) nên loại.
  SmoothGrad (N=8) gần như không khác baseline nên loại (đắt gấp 8 lần).
- Mới: `src/demo/saliency.py` (`select_salient_spans`, quantile 0.85),
  dùng chung trong `transformer_scorer.py` (M1/M2) + `hybrid_scorer.py` (M5),
  note evidence ghi rõ quy tắc lọc. M3/M4/B1 không đụng.
- Kiểm chứng: M1 evidence cặp C6 còn 8/9 vệt, 0 span chỉ-dấu-câu, tất định;
  `test_demo_scan` + `test_demo_server` xanh (sửa 1 test MOSS fallback cho khớp
  `_DEFAULT_USER_ID` mới). Không sinh lại hình báo cáo (M4/M5 cần Joern).
- Sinh lại hình 6.1/6.2 với logic saliency mới (máy có Joern tại
  `D:\Code\SDK\joern`, truyền qua env `JOERN_BIN` khi chạy):
  `workspace/demo_clone5.tex` + `demo_neg5.tex` viết lại, điểm số giữ nguyên
  khớp báo cáo (M4/M5 evidence chạy được, M1 clone còn 5/5 vệt).
  Build PDF 8 trang 0 error, copy sang `[10-10]` .pdf.
- Gọn + chính xác: thêm `filter_pairs_by_spans` (`saliency.py`) — M1/M2/M5 chỉ
  giữ cặp mutual-best chạm vùng salient (6.1: 30 cặp → M1 còn 2, M2 còn 4,
  M5 còn 3–4; cặp top-8 xếp theo saliency vẫn lẫn boilerplate `#include` nên loại).
  Sinh lại hình + build PDF 0 error; chú thích "Cách đọc hình" cập nhật theo.
- Theo yêu cầu: revert toàn bộ về cách ban đầu (mean + đủ pairs):
  `git checkout` transformer/hybrid scorer + project_structure, xóa `saliency.py`,
  chú thích báo cáo về cũ, sinh lại hình (M1 26/25, M2 30/30 vệt như cũ),
  build PDF 8 trang 0 error, test xanh. Giữ lại fix test MOSS fallback.

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

## [COMPLETE] M4 V2 + báo cáo (10/10)
- Train full 15 epoch: val F1 0,819 → **0,842** (E14), test F1 0,779 → **0,797**,
  ngưỡng mới 0,470. Train/val loss giảm mượt, không overfit.
- Chạy lại demo M4 mới (local, Joern): cặp clone 0,962 → **0,967**,
  cặp neg 0,294 → **0,190**. Tái sinh section CPG-GNN trong
  `workspace/demo_clone5.tex` + `demo_neg5.tex`.
- Báo cáo cập nhật: mục 1 (15 epoch), bảng train (hàng CPG mới + ghi tốt nhất E14),
  bảng test (0,723/0,887/0,797/0,882/0,470), câu 6.1/6.2. PDF build lại.

## [COMPLETE] Hybrid V2 -> verdict: KHÔNG tốt hơn, giữ V1 (10/10)
- V2 (3 epoch, LR nhỏ): val F1 0,914 -> 0,903 (kém), test F1 0,9133 -> 0,9138
  (hòa, trong nhiễu), ngưỡng sập 0,370 -> 0,010 (mỏng, đáng ngờ).
- Overfit vẫn còn (val E2 rớt 0,882 trong khi train loss vẫn giảm) — giảm LR
  chưa đủ, hướng tiếp theo là freeze encoder (chỉ train head).
- Khôi phục: `exp006_hybrid_poj104` = V1 (test 0,9133/thr 0,37), V2 cất sang
  `exp006_hybrid_poj104_v2`. Báo cáo giữ nguyên số V1, không cần sửa.

## [COMPLETE] Chuẩn bị train 512 token M1/M2 (warm-start từ 256)
- `transformer_pair_runner.py`: thêm `_maybe_warm_start()` — nạp `model_state`
  (strict, cùng kiến trúc, pos-embeds vốn đã 512), optimizer/scheduler reset.
  Chỉ kích hoạt khi config có `model.warm_start`, run cũ không ảnh hưởng.
- 2 config mới: `exp002_codebert_512.yaml` (M1) + `exp003_graphcodebert_512.yaml`
  (M2): `max_length` 256 → 512, `batch_size` 16 → 8, accum 2 → 4 (giữ
  effective batch 32), `pair_batch_size` 64 → 32, epochs 3 + patience 2.
- Checkpoint 256 (`best_weights_only.pt` M1/M2, ~506MB mỗi file) đã push lên HF
  `Kichirou24/SemanticCodeCloneDetection` (`m1/m2_pair_256_weights_only.pt`, LFS).
- 2 notebook mới `exp002_codebert_512_kaggle.ipynb` / `exp003_graphcodebert_512_kaggle.ipynb`
  (copy từ bản 256 + cell kéo weights HF về đúng path `warm_start`).
- Test: 33 passed (`test_configs` + `test_exp002` + `test_exp003`).
- Diagnostic trên Kaggle T4: M1 + M2 đều PASS (CUDA, gradient finite,
  params updated, loss giảm).

## [IN PROGRESS] Train M1/M2 512 trên Kaggle (user, warm-start từ 256)
1. Chạy 2 notebook 512 (M1 trước cho rẻ, rồi M2).
2. Khi xong: tải `results/exp00*_..._512/` về, so truncation mới
   (`truncation.json`, kỳ vọng rate rớt từ ~60-78% về gần 0) + val/test F1 vs bản 256.
3. Nếu ok mới xét M5-512 (text 512 + GNN, tốn VRAM nhất nên để sau cùng).

## [READY] M5 Hybrid-512 đấu M2-512 (config + notebook xong 13-10, chờ M2-512 về)
- `configs/exp006_hybrid_512.yaml`: max_length 512, warm-start text từ
  `results/exp003_graphcodebert_poj104_512/best_checkpoint.pt` + graph M4 giữ nguyên,
  batch 12->6 / accum 2->4 (effective ~24), pair_batch 32->16, LR V2
  (head 1e-5 / encoder 5e-6), out `results/exp006_hybrid_poj104_512`.
- `notebooks/exp006_hybrid_512_kaggle.ipynb`: copy từ bản 256, trỏ config/out mới,
  HF file `m2_512_text_encoder.pt` (+ `m4_graph_encoder.pt` giữ nguyên).
- Test: `test_configs` + `test_exp006_hybrid` + `test_demo_registry` xanh (15 passed).
- Việc còn lại (user): (1) export encoder M2-512 local
  (`export_hybrid_warm_start --m2 <m2-512 ckpt> --m4 <m4 ckpt>`), đổi tên
  `m2_text_encoder.pt -> m2_512_text_encoder.pt`, upload HF; (2) audit bổ trợ lại
  trên `test_scores.jsonl` M2-512 trước khi đốt ~8h train; (3) chạy notebook 512.

## [TO DO] Chạy trên Kaggle (user)
1. Import 2 notebook, `git pull` ở cell 2.
2. Chạy CPG trước (~3h), rồi Hybrid (~7-8h).
3. Backup `results/exp006_hybrid_poj104/best_checkpoint.pt` cũ nếu cần giữ số báo cáo 09-10
   (Hybrid train mới ghi đè thư mục kết quả).

## [PLAN] Sau khi có số mới
- Cập nhật bảng val F1 + test trong `reports/[09 - 10] - Báo cáo.tex`, build lại PDF.
- So sánh Hybrid V2 vs V1 (hết overfit E0 chưa), CPG epoch mới có tăng tiếp không.

## [BACKLOG] Tạm hoãn (vượt quota)
- ~~Nới giới hạn node CPG-GNN (attention O(n²)).~~ → đã đưa vào kế hoạch tuần sau (21-10).
- ~~Two-stage Hybrid: freeze encoder 1 epoch đầu rồi unfreeze (nếu V2 vẫn overfit).~~ → đã đưa vào kế hoạch tuần sau (19-10).
