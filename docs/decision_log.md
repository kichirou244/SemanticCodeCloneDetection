<a id="top"></a>

# Decision log (decision_log.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.10.0 |
| **Ngày cập nhật** | 2026-09-27 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [DEC-001 — Dataset](#dec-001--dataset)
- [DEC-002 — Model matrix](#dec-002--model-matrix)
- [DEC-003 — Split strategy](#dec-003--split-strategy)
- [DEC-004 — M4 backbone](#dec-004--m4-backbone)
- [DEC-005 — M5 fusion](#dec-005--m5-fusion)
- [DEC-006 — Docs layout](#dec-006--docs-layout)
- [DEC-007 — Bác split chính thức POJ-104](#dec-007--bac-split-chinh-thuc-poj-104)
- [DEC-008 — CodeNet audit bằng streaming](#dec-008--codenet-audit-bang-streaming)
- [DEC-009 — Thứ tự implement AST/CPG](#dec-009--thu-tu-implement-astcpg-khi-chua-co-joern)
- [DEC-010 — Dedup cross-problem trước split](#dec-010--dedup-cross-problem-truoc-split)
- [DEC-011 — Pair ratio và sampling caps](#dec-011--pair-ratio-va-sampling-caps)
- [DEC-012 — CodeNet status policy](#dec-012--codenet-status-policy)
- [DEC-013 — POJ-104 duplicate policy](#dec-013--poj-104-duplicate-policy)
- [DEC-014 — EXP-001 tokenization và threshold](#dec-014--exp-001-tokenization-va-threshold)
- [DEC-015 — EXP-002 CodeBERT checkpoint](#dec-015--exp-002-codebert-checkpoint)
- [DEC-016 — EXP-002 pair architecture và max_length](#dec-016--exp-002-pair-architecture-va-max-length)
- [DEC-017 — Sửa bug scheduler và device metadata của EXP-002](#dec-017--sua-bug-scheduler-va-device-metadata-cua-exp-002)
- [DEC-018 — Kaggle run protocol cho EXP-002](#dec-018--kaggle-run-protocol-cho-exp-002)
- [DEC-019 — Thu hẹp scope: chỉ C/C++ (POJ-104)](#dec-019--thu-hep-scope-chi-cc-poj-104)
- [Lịch sử thay đổi](#lich-su-thay-doi)

---

## DEC-001 — Dataset

- Quyết định: POJ-104 cho C/C++, CodeNet Python cho Python; duy nhất ở core.
- Lý do: đúng yêu cầu đề bài INIT; POJ-104 là benchmark clone C/C++ chuẩn,
  CodeNet Python phủ Python đa problem.
- Loại bỏ: OJClone, BigCloneBench, dataset C/C++ khác (ngoài core scope).

[🔝 Quay lại mục lục](#top)

---

## DEC-002 — Model matrix

- Quyết định: 1 baseline (TF-IDF+Cosine) + 5 core (CodeBERT, GraphCodeBERT,
  ASTNN, CPG-GNN, Hybrid). Không AST Similarity riêng, không UniXcoder/AST-GCN.
- Lý do: đủ trả lời RQ1–RQ3 với chi phí thí nghiệm kiểm soát được.

[🔝 Quay lại mục lục](#top)

---

## DEC-003 — Split strategy

- Quyết định: ưu tiên split chính thức nếu không leakage; fallback split theo
  problem identity (80/10/10, seed 42), implement ở
  `src/data/splitting/problem_split.py`.
- Lý do: ngăn cùng problem xuất hiện ở train/test.

[🔝 Quay lại mục lục](#top)

---

## DEC-004 — M4 backbone

- Quyết định: backbone chính mặc định là **GAT** (`configs/models/cpg_gnn.yaml`).
- Lý do: cần một cấu hình chính cho core experiment theo yêu cầu INIT.
- Mở: chốt cuối sau feasibility; đổi backbone phải thêm DEC mới.

[🔝 Quay lại mục lục](#top)

---

## DEC-005 — M5 fusion

- Quyết định: fusion INIT là **concat + MLP** (`configs/models/hybrid.yaml`).
- Lý do: đơn giản, đủ kiểm tra tính bổ sung thông tin; cross-attention để sau.

[🔝 Quay lại mục lục](#top)

---

## DEC-006 — Docs layout

- Quyết định: tài liệu nghiên cứu đặt trực tiếp dưới `docs/*.md` theo yêu cầu
  INIT của project, thay vì chỉ trong `docs/specs/...` của framework có sẵn.
- Lý do: framework `.agents/` là generic; research cần layout paper-ready riêng.
  Ghi nhận偏差 này để agent sau không "chuẩn hóa" ngược lại.

[🔝 Quay lại mục lục](#top)

---

## DEC-007 — Bác split chính thức POJ-104

- Quyết định: **không dùng split train/validation/test chính thức** của
  `google/code_x_glue_cc_clone_detection_poj104` cho core experiment; bắt buộc
  `problem_split` của project.
- Bằng chứng (`reports/004`, chạy 2026-09-20): official split có problem
  overlap (train×val 1, val×test 1) và exact code overlap (train×val 500,
  val×test 500) → `passed: false`. Trong khi `problem_split` passed true.
- Lý do: 500 mẫu giống hệt nhau xuyên split làm test metric bị thổi phồng,
  vi phạm fairness và là điểm phê bình khi review paper.
- Việc còn lại: điều tra problem ID cụ thể gây overlap ở phase sau.

[🔝 Quay lại mục lục](#top)

---

## DEC-008 — CodeNet audit bằng streaming

- Quyết định: audit CodeNet Python bằng `streaming + shuffle buffer + mẫu lớn`
  thay vì load full non-streaming trên Windows.
- Lý do: load full 4.4M mẫu crash khi build arrow cache
  (`OSError Errno 22` + `WinError 1450`) trên máy user; tham số mới
  (`--split/--streaming/--shuffle-buffer/--max-samples`) đã thêm vào loader và
  2 script. Audit full để phase sau (máy khỏe hơn hoặc cache sang ổ D).
- Hạn chế: ước lượng từ mẫu, không thay audit full; ghi rõ trong report.

[🔝 Quay lại mục lục](#top)

---

## DEC-009 — Thứ tự implement AST/CPG khi chưa có Joern

- Quyết định: AST extractor trước cho cả C++ và Python (grammars đã có) →
  CPG-Python (ast + CFG tự xây) → CPG-C++ sau khi cài Joern. Không thay M4
  bằng model khác.
- Lý do: `joern` chưa cài trên máy user (2026-09-20); cài cần Java + dung
  lượng thêm, trong khi M3 (ASTNN) và nửa M4-Python làm được ngay. M4-C++
  là blocker duy nhất, đã ghi ở `reports/005`.
- Việc còn lại: user quyết định cài Joern ở phase sau.

[🔝 Quay lại mục lục](#top)

---

## DEC-010 — Dedup cross-problem trước split

- Quyết định: dedup normalized toàn cục **trước** problem split, ít nhất cho
  CodeNet Python; POJ-104 kiểm tra lại ở phase pair generation.
- Bằng chứng (`reports/004`, 2026-09-20): problem split trên mẫu CodeNet 20k
  vẫn có exact code overlap train×test = 3 — cùng source xuất hiện ở problems
  khác nhau, nên split theo problem **không đủ**.
- Lý do: code trùng xuyên split làm rò rỉ trực tiếp đáp án vào test.
- Việc còn lại: implement dedup toàn cục (giữ 1 bản + ghi nguồn) ở phase pair
  generation; đo lại leakage sau dedup.

[🔝 Quay lại mục lục](#top)

---

## DEC-011 — Pair ratio và sampling caps

- Quyết định: tỉ lệ positive:negative = 1:1; cap positive/pair
  (POJ-104: train 2000, val/test 500; CodeNet pilot: train 500, val/test 200);
  negative random theo problem; problems 65/81 không có ưu tiên riêng ngoài cap
  chung (stratification bằng cap).
- Lý do: tránh accuracy ảo do imbalance và problem lớn dominate; giữ test
  không oversample. Xem `configs/pairs/*.yaml`, `src/data/pairs/generator.py`.

[🔝 Quay lại mục lục](#top)

---

## DEC-012 — CodeNet status policy

- Quyết định: **mọi build CodeNet áp `status_allowlist=["Accepted"]`**
  (đã enforce trong `configs/pairs/codenet_python.yaml`); toàn bộ experiment
  protocol ghi rõ CodeNet chỉ dùng accepted submissions.
- Bằng chứng: audit 50k split small = 100% Accepted, nhưng 50k split big chỉ
  ~60% Accepted (Wrong Answer ~20%, Runtime Error ~14%, TLE/MLE/OLE còn lại).
- Lý do: rejected submissions có thể sai chức năng, làm nhiễu nhãn positive.
  Pilot small không bị ảnh hưởng (đã thuần accepted) nhưng rebuild lại cho
  manifest/config nhất quán.

[🔝 Quay lại mục lục](#top)

---

## DEC-013 — POJ-104 duplicate policy

- Quyết định: loại exact + normalized duplicates, giữ canonical = sample_id
  nhỏ nhất (deterministic); ghi mapping `duplicate_id → canonical_id`.
- Lý do: ~1.9% exact dup POJ-104; positive pairs phải là semantic clones thật,
  không phải bản copy. Không cherry-pick: mọi bản loại đều ghi trong manifest.

[🔝 Quay lại mục lục](#top)

---

## DEC-014 — EXP-001 tokenization và threshold

- Quyết định: TF-IDF dùng regex `[A-Za-z_][A-Za-z0-9_]*|\S`, giữ nguyên case
  (`lowercase=False`), không xử lý comment, không rename identifier;
  threshold chọn trên validation tối ưu F1 (grid 0.01), freeze rồi đánh giá
  test đúng 1 lần.
- Lý do: baseline phải là lexical thuần túy, độc lập M1–M5; chọn threshold
  trên test là leakage phương pháp luận.

[🔝 Quay lại mục lục](#top)

---

## DEC-015 — EXP-002 CodeBERT checkpoint

- Quyết định: M1 dùng `microsoft/codebert-base` (reuse identifier đã chốt ở
  `configs/models/codebert.yaml` và feasibility INIT). Không UniXcoder/CodeT5/LLM khác.
- Lý do: đúng scope M1; checkpoint đã verify tải được ở smoke INIT.

[🔝 Quay lại mục lục](#top)

---

## DEC-016 — EXP-002 pair architecture và max_length

- Quyết định: siamese shared encoder, pair `[hA; hB; |hA-hB|]` + MLP 2 lớp
  (Tanh, dropout 0.1); `max_length=256` mỗi phía, dynamic padding.
- Lý do: một architecture duy nhất cho EXP-002 (không thử nhiều kiến trúc);
  256 tokens/phía đủ phủ median code POJ (~150 tokens), giảm OOM so với 512;
  truncation rate đo và report ở `truncation.json`.
- Training mặc định: epochs 3, batch 16, accum 2 (effective 32), lr 2e-5,
  warmup 0.1, AMP, early stopping val-F1 patience 2. Freeze sau preflight.

[🔝 Quay lại mục lục](#top)

---

## DEC-017 — EXP-002 scheduler + device audit fix

- Quyết định: scheduler warmup-linear phải `step()` mỗi optimizer update
  (không phải mỗi epoch); `steps_per_epoch` tính bằng ceil; thêm FAIL-FAST
  device asserts (`assert_device_setup`, assert input/label/logit device) và
  metadata runtime thực (`device_requested/device_actual/model_device/cuda +
  gpu mem`); thêm `--mode diagnostic`; run cũ archive thành
  `results/exp002_codebert_poj104_invalid_run/` (INVALID, không phải M1 result).
- Lý do: audit 2026-09-21 (reports/010_codebert_audit.md) phát hiện F-01 scheduler
  per-epoch khiến lr thực ~1e-8 (thay vì 2e-5) → classifier collapse
  (test TN=0, ROC-AUC 0.52); F-02 metadata hardcode `device=cpu` gây hiểu lầm.
  Pair protocol (sample_id prefix shard) xác minh không leakage.
- Không đổi: architecture DEC-016, max_length 256, threshold-trên-val, dataset.

[🔝 Quay lại mục lục](#top)

---

## DEC-017 — Sửa bug scheduler và device metadata của EXP-002

- Quyết định: `get_linear_schedule_with_warmup` step theo **từng optimizer update**
  (không theo epoch); metadata runtime lấy từ `torch.cuda` thật
  (`collect_device_metadata` + `assert_device_setup` + assert device mỗi batch);
  thêm mode `diagnostic`; bỏ `device=cpu` hardcode.
- Lý do: audit 2026-09-21 (reports/010_codebert_audit.md) phát hiện F-01 scheduler
  per-epoch khiến lr thực ~1e-8 (thay vì 2e-5) → classifier collapse
  (test TN=0, ROC-AUC 0.52); F-02 metadata hardcode `device=cpu` gây hiểu lầm.
  Pair protocol (sample_id prefix shard) xác minh không leakage.
- Không đổi: architecture DEC-016, max_length 256, threshold-trên-val, dataset.

[🔝 Quay lại mục lục](#top)

---

## DEC-018 — Kaggle run protocol cho EXP-002

- Bối cảnh: EXP-002 cần ~7.8h train (đo được ở run invalid: 27,965s) trên khi
  máy local chỉ phù hợp smoke test. Kaggle giới hạn **12h/phiên** và kill không báo
  trước. Trước DEC-018, `training_history.csv` + `run_metadata.json` chỉ ghi
  **sau khi hết cả 3 epoch** → session chết ở giờ thứ 10 là mất sạch 8h.
- Quyết định:
  1. **Flush artifact mỗi epoch** (atomic write qua `.tmp` + `replace`):
     `training_history.csv`, `run_metadata.json`, `last_checkpoint.pt`,
     `best_checkpoint.pt`.
  2. **`--time-budget-s`** (mặc định 9h) dừng train graceful rồi vẫn chạy
     evaluate, thay vì bị Kaggle kill giữa chừng.
  3. **`--mode resume`** nạp `last_checkpoint.pt` (model + optimizer + scheduler +
     RNG state) để train tiếp epoch còn lại thay vì chạy lại từ đầu.
  4. **Bỏ checkpoint từng epoch**, chỉ giữ `best` + `last` (6GB → 3GB trong
     `/kaggle/working`).
  5. `EVAL_COMPLETE.json` làm marker để notebook biết chắc eval xong trước khi tar.
  6. Notebook dùng `subprocess` thay vì `!cmd $var` — IPython **không** nội suy
     biến Python trong `!`, `$url` bị bash expand thành rỗng nên `git clone` chết
     ngay cell 2. Cell 2 idempotent (pull nếu đã clone) để restart-resume được.
  7. Không `pip install` torch/transformers mặc định (Kaggle đã có bản CUDA tương
     thích); chỉ cài module còn thiếu.
  8. Smoke/diagnostic giới hạn `codes` theo refs thực dùng, bỏ phân tích
     truncation trên toàn bộ 41k codes (nhanh 56s thay vì >5 phút).
- Lý do: biến "1 lần chạy 8h không thể cứu" thành "chạy được nhiều lần, mỗi lần
  đều tạo ra artifact có giá trị". Bắt buộc cho mọi EXP sau (EXP-003…012) vì
  cùng bài toán thời gian dài.
- Xác minh: 39/39 test pass (5 test mới cho flush/resume/time-budget/marker/atomic);
  `--mode diagnostic --device cuda` → `DIAGNOSTIC: PASS` trên RTX 5060
  (loss 1.009 → 0.885, `model_on_cuda=true`, 126,416,641 params).
- Không đổi: hyperparameter (`configs/exp002_codebert.yaml` giữ nguyên), dataset,
  architecture, protocol train/val/test.

[🔝 Quay lại mục lục](#top)

---

## DEC-019 — Thu hẹp scope: chỉ C/C++ (POJ-104)

- Quyết định: bỏ khối thí nghiệm Python. EXP-007…EXP-012 chuyển `CANCELLED`.
  Toàn bộ kết quả báo cáo chỉ dựa trên POJ-104 (C/C++).
- Lý do:
  1. **RQ1–RQ4 không phụ thuộc ngôn ngữ.** Chúng so sánh representation
     (M1 vs M2 vs M3 vs M4), transformer-vs-program-graph (M2 vs M4), hybrid
     (M2 vs M4 vs M5) và robustness dưới transformation. POJ-104 trả lời
     đủ cả bốn.
  2. **Quota Kaggle không đủ.** EXP-008…012 dự kiến ~11.3h mỗi run (CodeNet có
     484,292 train pairs so với 332,000 của POJ-104, gấp 1.46×), tổng ~57h vượt
     quota 30h/tuần. POJ-104 còn ~20h, vừa một tuần.
  3. EXP-008 dự kiến vượt cả giới hạn 12h/phiên, buộc phải hạ `max_length` hoặc
     giảm epoch — tạo thêm biến số không cần thiết cho so sánh công bằng.
- Cái mất: không có bằng chứng cross-language generalization. Ghi rõ ở phần
  limitation/discussion thay vì bỏ trống — reviewer có thể hỏi, nên trả lời
  trước.
- Cái giữ nguyên: dataset CodeNet đã audit và build xong (`reports/003/006`,
  manifest `data/manifests/codenet_python.json`, 547k pairs, leakage passed) và
  code loader vẫn còn. Nếu sau này cần mở rộng sang Python thì dùng lại được
  ngay, không tốn thí nghiệm nào đã chạy.
- Ước lượng phần còn lại (POJ-104): EXP-003 ~7.8h, EXP-004 ~2h, EXP-005 ~2h,
  EXP-006 ~8h → **~20h**, khớp quota 1 tuần. Early stopping (patience 2) có thể
  cắt ngắn thêm.
- Ảnh hưởng tài liệu: `docs/research_questions.md` (thêm ghi chú scope),
  `docs/project_overview.md`, `README.md`, `docs/experiment_registry.md`
  (EXP-007…012 → CANCELLED).

[🔝 Quay lại mục lục](#top)

---

## Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | 6 quyết định INIT đầu tiên. |
| 2 | 1.1.0 | 2026-09-20 | Thêm DEC-007 (bác official split POJ-104), DEC-008 (CodeNet streaming). |
| 3 | 1.2.0 | 2026-09-20 | Thêm DEC-009 (thứ tự AST/CPG, Joern pending). |
| 4 | 1.3.0 | 2026-09-20 | Thêm DEC-010 (dedup cross-problem trước split). |
| 5 | 1.4.0 | 2026-09-20 | Thêm DEC-011 (pair ratio/caps), DEC-013 (POJ-104 dup policy). |
| 6 | 1.5.0 | 2026-09-20 | Thêm DEC-012 (CodeNet status: pilot nguyên trạng, big/full audit trước). |
| 7 | 1.6.0 | 2026-09-20 | Thêm DEC-014 (EXP-001 tokenization + threshold validation-F1). |
| 8 | 1.7.0 | 2026-09-20 | Thêm DEC-015 (CodeBERT checkpoint), DEC-016 (pair arch + max_length 256). |
| 9 | 1.8.0 | 2026-09-21 | Thêm DEC-017 (fix scheduler per-update + device metadata thật). |
| 10 | 1.9.0 | 2026-09-27 | Thêm DEC-018 (Kaggle run protocol: flush mỗi epoch, time budget, resume). |
| 11 | 1.10.0 | 2026-09-27 | Thêm DEC-019 (thu hẹp scope: chỉ C/C++ POJ-104, hủy EXP-007…012). |
