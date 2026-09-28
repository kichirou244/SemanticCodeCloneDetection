<a id="top"></a>

# Experiment registry (experiment_registry.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.10.0 |
| **Ngày cập nhật** | 2026-09-28 |
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
| EXP-001 | POJ-104 | B1 TF-IDF | COMPLETED | Test P/R/F1 0.552/0.860/0.673, thr 0.24 | reports/009_baseline_results.md |
| EXP-002 | POJ-104 | M1 CodeBERT | COMPLETED | Test P/R/F1 0.931/0.890/0.910, ROC-AUC 0.973, thr 0.59 (val-selected), best epoch 1/3, 7,03h T4 | reports/011_codebert_results.md |
| EXP-003 | POJ-104 | M2 GraphCodeBERT | READY_FOR_KAGGLE_TRAIN | Implementation xong, dùng **chung code path** với EXP-002 (DEC-020); 12 test khóa tính công bằng (13 test gồm cả notebook EXP-003 phải mirror EXP-002); notebook `notebooks/exp003_graphcodebert_kaggle.ipynb` sẵn sàng; throughput đo = EXP-002 ⇒ **~7,8h** | - |
| EXP-004 | POJ-104 | M3 ASTNN | READY_FOR_KAGGLE_TRAIN | Implementation xong (DEC-021): extractor thật, RvNN vectorize, cache AST; 30 test; preflight + export bản nhẹ PASS trên RTX 5060; notebook `notebooks/exp004_astnn_kaggle.ipynb` sẵn sàng; **~35 phút** trên T4 (hiệu chuẩn, cần verify trực tiếp) | - |
| EXP-005 | POJ-104 | M4 CPG-GNN | BLOCKED_ON_JOERN | Schema + ingest + store + model GAT (204.801 tham số) **đã xong, 33 test local không cần Joern** (DEC-024); còn thiếu CPG thật — cài Joern **trên Kaggle** (`notebooks/exp005_cpg_extract_kaggle.ipynb`), đo 500 hàm trước khi cam kết 5–17h | - |
| EXP-006 | POJ-104 | M5 Hybrid | PLANNED | - | - |
| EXP-007 | CodeNet Python | B1 TF-IDF | CANCELLED | - | DEC-019 |
| EXP-008 | CodeNet Python | M1 CodeBERT | CANCELLED | - | DEC-019 |
| EXP-009 | CodeNet Python | M2 GraphCodeBERT | CANCELLED | - | DEC-019 |
| EXP-010 | CodeNet Python | M3 ASTNN | CANCELLED | - | DEC-019 |
| EXP-011 | CodeNet Python | M4 CPG-GNN | CANCELLED | - | DEC-019 |
| EXP-012 | CodeNet Python | M5 Hybrid | CANCELLED | - | DEC-019 |

Trạng thái: PLANNED / RUNNING / COMPLETED / FAILED / BLOCKED / INVALIDATED / CANCELLED.

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
