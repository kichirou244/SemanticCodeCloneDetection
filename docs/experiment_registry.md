<a id="top"></a>

# Experiment registry (experiment_registry.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.17.0 |
| **Ngày cập nhật** | 2026-10-17 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Registry](#1-registry)
- [2. Lịch sử thay đổi](#2-lich-su-thay-doi)

---

## 1. Registry

| Experiment | Dataset | Model | Status | Result | Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 | POJ-104 | B1 TF-IDF | COMPLETED (tham khảo) | Test P/R/F1 0.552/0.860/0.673, thr 0.24. **Không dùng làm baseline chính** — là mô hình bag-of-token của chính nhóm, so với nó thì bất công với mô hình học sâu | reports/009_baseline_results.md |
| EXP-014 | POJ-104 | B1 MOSS service thật (mosspy) | COMPLETED | **Baseline chính của báo cáo.** Gọi thẳng `moss.stanford.edu` (user id qua env `MOSS_USER_ID`), rule vận hành `score > 0.01` ⇒ clone. Test P 1.0 / R 0.0016 / F1 0.0033 (±0.001), flag 9/11000 (9/5500 positive); val 0 flag. **Top-N bias của chính server**: chỉ trả 1000 match mạnh nhất (`show=1000`) nên không chấm điểm được cặp semantic clone Type-3/4 ⇒ phần lớn positive bị điểm 0; ROC-AUC không tính được. Report: val `http://moss.stanford.edu/results/6/3324282961870`, test `http://moss.stanford.edu/results/7/5556361909376` (cache `artifacts/moss_cache` — chạy lại không tốn lượt). Chú ý rate-limit khi nộp | scripts/eval_moss_service.py |
| EXP-013 | POJ-104 | B0 MOSS (Winnowing) | COMPLETED (tham khảo) | Bản winnowing all-pairs **local**, giữ làm tham chiếu + fallback demo (k=20,w=5). Test P/R/F1 0.646/0.722/0.682, ROC-AUC 0.731, thr chọn trên validation. Chứng minh MOSS đạt F1 cao nếu chấm được *mọi* cặp — nhưng đó không phải cách service thật hoạt động (chỉ top-N) | scripts/eval_moss.py |
| EXP-002 | POJ-104 | M1 CodeBERT | COMPLETED | Test P/R/F1 0.931/0.890/0.910 ± 0.003, ROC-AUC 0.973, thr 0.59 (val-selected), best epoch 1/3, 7,03h T4 | reports/011_codebert_results.md |
| EXP-003 | POJ-104 | M2 GraphCodeBERT | COMPLETED | Test P/R/F1 0.883/0.933/0.907 ± 0.003, ROC-AUC 0.971, thr 0.43 (val-selected); 3 epoch, val F1 **còn đang tăng** ⇒ số có thể còn tăng | reports/012_m1_m2_m3_summary.pdf |
| EXP-004 | POJ-104 | M3 ASTNN | COMPLETED | Test P/R/F1 0.760/0.894/0.822 ± 0.004, ROC-AUC 0.889, thr 0.42 (val-selected); train tiếp trần 20 epoch, early stopping dừng ở epoch 7 (tốt nhất epoch 4: val F1 0.853), ~40 phút T4 | reports/012_m1_m2_m3_summary.pdf |
| EXP-005 | POJ-104 | M4 CPG-GNN | COMPLETED | Test P/R/F1 0.695/0.885/0.779 ± 0.004, ROC-AUC 0.864, thr 0.42 (val-selected); 5 epoch (~57 phút T4), val F1 còn tăng tới epoch cuối (chưa plateau); kém CodeBERT 0,131 (CI không chứa 0), hơn MOSS 0,097 | - |
| EXP-006 | POJ-104 | M5 Hybrid (GraphCodeBERT + CPG-GNN) | READY_FOR_KAGGLE_TRAIN | Code + model + config + **`notebooks/exp006_hybrid_kaggle.ipynb`** + 5 test xong (DEC-029): soft fusion embedding concat_mlp, warm start encoder M2+M4, joint fine-tune LR 2e-5. **Baseline logistic (s_M2,s_M4): val F1 0.9463, test F1 0.9217** (M2 0.9069, +1.5đ) nhưng AUC 0.9683 < M2 0.9713 ⇒ chỉ thắng vùng ngưỡng ~0.535, phải nêu đúng khi báo cáo. Audit bổ trợ: agreement 0.724, kappa 0.44; only_m4 662 (sửa 62.9% lỗi M2) nhưng hỏng 2377; OR/AND bị loại (F1 0.809/0.879 < M2) | notebooks/exp006_hybrid_kaggle.ipynb, scripts/audit_m2_m4_complementarity.py, results/audit_m2_m4.json |
| EXP-007 | CodeNet Python | B1 TF-IDF | CANCELLED | - | DEC-019 |
| EXP-008 | CodeNet Python | M1 CodeBERT | CANCELLED | - | DEC-019 |
| EXP-009 | CodeNet Python | M2 GraphCodeBERT | CANCELLED | - | DEC-019 |
| EXP-010 | CodeNet Python | M3 ASTNN | CANCELLED | - | DEC-019 |
| EXP-011 | CodeNet Python | M4 CPG-GNN | CANCELLED | - | DEC-019 |
| EXP-012 | CodeNet Python | M5 Hybrid | CANCELLED | - | DEC-019 |
| EXP-FINAL-001 | POJ-104 | Eval + chọn + inference (10 run) | COMPLETED | Xếp hạng theo val F1: **M5-512 thắng** (0,9379) hơn M2-512 (0,9337) chỉ +0,0043; McNemar val p=0,0814 (không ý nghĩa) vs paired bootstrap CI95 [+0,0004,+0,0083] (bất đồng ⇒ `mixed`); ngưỡng M5-512 0,010 rất mỏng; M2-512 hơn ở test F1 0,9550/AUC 0,9912 ⇒ **pipeline cuối dùng M2-512 cho `detect`** (`inference.default_model`). `select` không cần GPU | reports/014_final_selection.md, `results/final_pipeline/` |

Trạng thái: PLANNED / RUNNING / COMPLETED / FAILED / BLOCKED / INVALIDATED / CANCELLED.

**EXP-FINAL-001 (pipeline cuối) — 2026-10-17, DEC-030.** Không train thêm. Đọc artifact
đã đóng băng của 10 run (baseline + candidate) → xếp hạng theo **val F1** → kiểm định cặp
winner vs á quân trên validation (McNemar + paired bootstrap) → ghi
`results/final_pipeline/{leaderboard.json,leaderboard.md,selection.json}`. Cấu hình
`configs/final_pipeline.yaml`, code `src/evaluation/leaderboard.py` +
`scripts/final_pipeline.py` (`select` chạy trên CPU, `detect` chấm một cặp file bằng model
đã chọn). Xem `reports/014_final_selection.md` và `docs/evaluation_protocol.md` §4.

**Scope: chỉ C/C++ (POJ-104)** theo DEC-019. EXP-007…012 hủy vì (a) RQ1–RQ4 không
phụ thuộc ngôn ngữ — POJ-104 trả lời đủ; (b) EXP-008…012 dự kiến ~11.3h mỗi run
(CodeNet 484,292 train pairs = 1.46× POJ-104) → vượt quota Kaggle 30h/tuần.
Dataset CodeNet đã audit + build xong và vẫn giữ trong repo, mở rộng lại được
ngày nào cần mà không tốn thí nghiệm đã chạy.

Ghi chú Phase 1 (2026-09-20): pipeline dataset chung đã implement
(`scripts/build_dataset.py` + manifests); config EXP-001 sẵn ở
`configs/experiments/exp001.yaml`. **POJ-104 pairs built + leakage passed
(353k pairs, manifest `data/manifests/poj104.json`) — sẵn sàng EXP-001.
CodeNet pilot finalized (547k pairs, accepted-allowlist, passed).**
EXP-001/EXP-007 vẫn PLANNED cho tới khi chạy model thật.

Ghi chú EXP-002 (2026-09-28): **đã train xong trên Kaggle** (3/3 epoch, 7,03h, Tesla T4,
`device_actual: cuda:0` trong `run_metadata.json` — bằng chứng fix F-02 của DEC-017 chạy
thật) và **đã evaluate** ⇒ test F1 **0.910** (ROC-AUC 0.973), vượt EXP-001 **+0.237 F1**.
Báo cáo: `reports/011_codebert_results.md`. Ba điều chưa xong, ghi lại để không quên:
(1) `max_length 256` cắt mất 78,6 % code train — lỗi tập trung ở quartile dài nhất
(12,4 % vs 5,2 %), **đóng băng** cho M1/M2 theo DEC-022, ablation để sau; (2)
`early_stopping` không có `min_delta` nên không bao giờ kích hoạt, lãng phí ~2,3 h ở
epoch 3; (3) checkpoint 1,44 GB không commit (`.gitignore` loại `*.pt`) — cần
`last_checkpoint.pt` nếu muốn resume.
**Evaluate chạy được ở mọi nơi** (`--mode evaluate`, 5 phút) nên từ giờ chỉ train mới cần
quota GPU dài; theo quyết định mới nhất, notebook chạy evaluate **ngay trên Kaggle** rồi
tải metrics + predictions về — xem DEC-022 §2 và DEC-023.

Ghi chú EXP-003/EXP-004 (2026-09-27, cập nhật 2026-09-28): cả hai đã implement và test
xong, **chưa train lần nào** — status `READY_FOR_KAGGLE_TRAIN` nghĩa là "code + test +
preflight sẵn sàng", không phải đã có số liệu.

**Thứ tự chạy: EXP-004 → EXP-003** (DEC-023), mỗi mô hình một notebook riêng:
`notebooks/exp004_astnn_kaggle.ipynb` (~35 phút) và `notebooks/exp003_graphcodebert_kaggle.ipynb`
(~7,8h). M3 phải chạy trước vì **rẻ hơn 13×** — nếu phiên M2 chết giữa chừng thì vẫn giữ
được M3. Cả hai nằm trong một ngày quota 30h GPU (tổng ~8,4h < giới hạn 12h/phiên).
(Trước đó ghi "EXP-003 → EXP-004" vì EXP-002 là baseline trung lập — nay đảo lại vì lý do
phụ thuộc, không phải vì EXP-002 thiếu.)

**Evaluate chạy trên Kaggle**, mỗi notebook có cell `--mode evaluate` riêng (~5 phút M2,
~1 phút M3) rồi đóng gói **chỉ metrics + predictions** (vài MB) — không tải checkpoint
1,44GB về chạy lại local. Bật `KEEP_WEIGHTS = True` nếu muốn giữ `best_weights_only.pt`.

**EXP-005 (M4 CPG-GNN) — phần code đã xong, phần dữ liệu thì chưa** (DEC-024/025).
Đã có: schema CPG đóng băng (13 nhãn node / 9 nhãn cạnh), `ingest` (JSON → `CPGraph`),
`CPGStore` (cache `.npz`), model `CPGGNNPairClassifier` (GAT 3 tầp × 4 head, 204.801
tham số, head giống M1/M2/M3) — **34 test xanh local, không cần Java/GPU**.
Chưa có: CPG thật, vì Joern cần JVM và máy local không có Java ⇒ **cài Joern trên
Kaggle** qua `notebooks/exp005_cpg_extract_kaggle.ipynb`, chạy **cell smoke** (in
`cpg.help` + probe từng step CPGQL) trước, rồi đo 20 → 500 hàm để lấy `ham/giây`
thật trước khi cam kết 5–17 h cho 62k hàm. Exporter là **script Scala**
(`scripts/joern/export_cpg.sc`, Joern v4) và gom cả lô vào một tiến trình.
Config `configs/exp005_cpg_gnn.yaml` mang `status: BLOCKED_ON_JOERN` — xem
`reports/005_ast_cpg_feasibility.md`.

[🔝 Quay lại mục lục](#top)

---

## 2. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo 12 experiment core (6 POJ-104 + 6 CodeNet Python). |
| 2 | 1.1.0 | 2026-09-20 | Ghi chú Phase 1: pipeline dataset + config EXP-001 sẵn, status giữ PLANNED. |
| 3 | 1.2.0 | 2026-09-20 | POJ-104 pairs built + passed (sẵn sàng EXP-001); CodeNet pilot chờ rebuild. |
| 4 | 1.3.0 | 2026-09-20 | EXP-001 COMPLETED (test F1 0.673; report 009); EXP-002 READY_FOR_GPU (config + runner + tests xong). |
| 5 | 1.4.0 | 2026-09-27 | EXP-002 → READY_FOR_KAGGLE_TRAIN (DEC-018 Kaggle run protocol + diagnostic GPU PASS). |
| 6 | 1.5.0 | 2026-09-27 | DEC-019: thu hẹp scope chỉ C/C++ — EXP-007…012 → CANCELLED. |
| 7 | 1.6.0 | 2026-09-27 | EXP-003/EXP-004 → READY_FOR_KAGGLE_TRAIN (M2 dùng chung code path M1 theo DEC-020; M3 ASTNN implement xong theo DEC-021, ~35 phút). Sửa dòng lịch sử bị trùng. |
| 8 | 1.7.0 | 2026-09-28 | EXP-002 → COMPLETED: test F1 0.910 (ROC-AUC 0.973), +0.237 so với EXP-001; report `011_codebert_results.md`; thêm DEC-022 (đóng băng `max_length 256` + chưa thêm `min_delta` cho M1/M2, eval chạy local). |
| 9 | 1.8.0 | 2026-09-28 | EXP-003 có notebook Kaggle `notebooks/exp003_graphcodebert_kaggle.ipynb` (mirror EXP-002, có test chặn lệch); luồng tải kết quả chuyển sang checkpoint bản nhẹ `best_weights_only.pt` (0,5GB) — DEC-022 §2b. |
| 11 | 1.10.0 | 2026-09-28 | EXP-005 chuyển `PLANNED` → `BLOCKED_ON_JOERN`: đã có schema/ingest/store/model GAT (33 test local, không cần Joern), Joern cài **trên Kaggle**, chờ số đo `ham/giây` thật — DEC-024. |
| 10 | 1.9.0 | 2026-09-28 | Tách notebook: EXP-004 có notebook riêng `notebooks/exp004_astnn_kaggle.ipynb`; bỏ bản gộp 1 phiên. Mọi notebook có cell `--mode evaluate` chạy trên Kaggle và chỉ tải metrics + predictions (`KEEP_WEIGHTS=False`) — DEC-023. |
| 12 | 1.11.0 | 2026-10-07 | EXP-014 COMPLETED: **baseline chính chuyển sang MOSS service thật** (mosspy, `moss.stanford.edu`), rule `score > 0.01`. EXP-013 (winnowing all-pairs local) hạ xuống tham khảo/fallback. Lý do: service MOSS chỉ trả top-N match nên không chấm điểm được cặp semantic clone type-3/4 trong bộ cặp cố định (test chỉ bắt 9/5500 dương, F1 0.003). Điều này **không phải lỗi cài đặt**: `_safe_name` khớp 100% sid; bản local cùng thuật toán vẫn đạt F1 0.682 khi chấm điểm mọi cặp. Demo B1 = service nếu có `MOSS_USER_ID`, ngược lại fallback k=20,w=5. Report + demo đã cập nhật theo EXP-014. |
| 13 | 1.12.0 | 2026-10-07 | EXP-006 → **READY_FOR_KAGGLE_TRAIN** (M5 Hybrid, DEC-029): `configs/exp006_hybrid.yaml`, `src/models/hybrid/model.py`, `src/experiments/exp006_hybrid.py`, `notebooks/exp006_hybrid_kaggle.ipynb`, `tests/test_exp006_hybrid.py` (5 test, không GPU/HF). Chốt hướng: **soft fusion mức embedding** (concat [hA;hB;\|hA-hB\|] → MLP, DEC-005 + DEC-016), **không OR/AND** — audit `results/audit_m2_m4.json` cho OR F1 0.809 / AND F1 0.879 đều kém M2 0.907; logistic baseline test F1 0.9217 (+1.5đ vs M2) nhưng AUC 0.9683 < 0.9713 ⇒ chỉ thắng vùng ngưỡng. Warm start encoder từ EXP-003 + EXP-005; giữ max_length 256. |
| 14 | 1.13.0 | 2026-10-07 | **Kênh phân phối checkpoint warm start của EXP-006 chuyển sang Hugging Face Hub** (DEC-029 §4 bổ sung): checkpoint `*.pt` bị `.gitignore` nên không thể nằm trong repo; EXP-003 output trên Kaggle mặc định `KEEP_WEIGHTS=False` nên **không** chứa `best_checkpoint.pt` 1.41GB. Thêm `scripts/export_hybrid_warm_start.py` (rút riêng `encoder.*`: M2 475MB / 199 khóa, M4 0.4MB / 32 khóa — đã verify warm start M4 missing=0) và đẩy lên HF repo `Kichirou24/SemanticCodeCloneDetection`. Notebook `exp006_hybrid_kaggle.ipynb` cell 7 **pull từ HF** (`HF_TOKEN` nếu repo private), kèm fallback quét `/kaggle/input` cho output EXP-003/EXP-005 cũ; thiếu ⇒ assert chặn train. |
| 15 | 1.14.0 | 2026-10-07 | Chốt tên HF repo chính thức: `Kichirou24/SemanticCodeCloneDetection` (public) — notebook cell 7 + `export_hybrid_warm_start.py` đổi cho khớp; 2 file encoder (`m2_text_encoder.pt` 475MB, `m4_graph_encoder.pt` 0.4MB) đã push lên HF kèm README model card. |
| 16 | 1.15.0 | 2026-10-07 | **Errata row 15:** repo HF đang để **private** (anonymous 401), không phải public. Push/thao tác local qua **SSH** `git@hf.co` (đã đẩy 2 commit README+weights thành công). Trên Kaggle pull được nhờ `HF_TOKEN` của chính acc (cell 1 → cell 7); README HF ghi rõ điều này. |
| 17 | 1.16.0 | 2026-10-07 | Chay thử EXP-006 trên Kaggle: cell 7 báo 404 Repository Not Found dù repo vừa push qua SSH thành công. Kết luận vận hành: HF **trả 404 (không phải 401)** khi account của token **không sở hữu/không có quyền** truy cập repo private — che giấu sự tồn tại của repo. Cell 7 được nâng cấp: in whoami() để lộ account của token ngay khi pull lỗi. Khuyến nghị: để repo **public** (không cần token) hoặc dùng token đúng acc sở hữu. |
| 18 | 1.17.0 | 2026-10-17 | Thêm **EXP-FINAL-001** (DEC-030): pipeline cuối đánh giá + chọn + inference. Xếp hạng 10 run theo val F1 → M5-512 nhỉnh nhất nhưng +0,0043 so M2-512, McNemar p=0,0814 vs bootstrap CI chạm 0 (`mixed`), ngưỡng 0,010 mỏng, thua test F1/AUC ⇒ **pipeline cuối dùng M2-512 cho `detect`**. Artifact `results/final_pipeline/`; report `reports/014_final_selection.md`. |
