<a id="top"></a>

# Decision log (decision_log.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.30.0 |
| **Ngày cập nhật** | 2026-10-17 |
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
- [DEC-020 — M2 dùng chung code path với M1](#dec-020--m2-dung-chung-code-path-voi-m1-khong-clone-runner)
- [DEC-021 — M3 ASTNN: vectorize, cache, và đo thay vì đoán thời gian](#dec-021--m3-astnn-vectorize-rvnn-cache-cay-ast-v--%C4%91o-thay-v%C4%AB-%C4%91o%C3%A1n-th%C4%9Di-gian)
- [DEC-022 — Đóng băng recipe M1/M2 + evaluate chạy local](#dec-022--%C4%91%C3%B3ng-b%C4%91ng-recipe-m1m2-sau-run-exp-002-evaluate-ch%E1%BA%A1y-local)
- [DEC-023 — Notebook riêng cho M3 và M2, chạy M3 trước; evaluate ở Kaggle; EXP-005 là blocker](#dec-023--notebook-ri%C3%AEng-cho-m3-v%C3%A0-m2-ch%E1%BA%A1y-m3-tr%C6%B0%E1%BB%9Bevaluate-%E1%BB%9F-kaggleexp-005-l%C3%A0-blocker)
- [DEC-024 — M4 contract-first: đo thay vì cài, Joern chạy trên Kaggle không cài local](#dec-024--m4-contract-first--%C4%91o-thay-v%C3%AC-c%C3%A3i-joern-ch%E1%BA%A1y-tr%C3%AAn-kaggle-kh%C3%B4ng-c%C3%A1i-local)
- [DEC-025 — Joern v4: script Scala + gom cả lô vào một tiến trình, và smoke test bắt buộc](#dec-025--joern-v4-script-scala--gom-c%E1%BA%A3-l%C3%B4-v%C3%A0o-m%E1%BB%99t-ti%E1%BA%BFn-tr%C3%ACnh-v%C3%A0-smoke-test-b%E1%BA%AFt-bu%E1%BB%99c)
- [DEC-026 — Baseline của báo cáo là MOSS (EXP-013), không dùng TF-IDF](#dec-026--baseline-c%E1%BB%A3a-b%C3%A1o-c%C3%A1o-l%C3%A1-mo%E1%BB%85-013-kh%C3%B4ng-d%C3%B9ng-tf-idf)
- [DEC-027 — M4 schema (b): bỏ CALL/REF/REACHING_DEF, sửa PARAMETER_OUT bằng khớp tên](#dec-027--m4-schema-b-b%E1%BB%8F-callrefreaching_def-s%E1%BB%ADa-parameter_out-b%E1%BA%B1ng-kh%E1%BB%9Bp-t%C3%AAn)
- [DEC-028 — Baseline chính chuyển sang MOSS service thật (EXP-014); EXP-013 còn lại làm tham chiếu](#dec-028--baseline-ch%C3%ADnh-chuy%E1%BB%83n-sang-moss-service-th%E1%BA%ADt-exp-014-exp-013-c%C3%B2n-l%E1%BA%A1i-l%C3%A0m-tham-chi%E1%BA%BFu)
- [DEC-029 — M5 hybrid soft fusion (EXP-006): fusion mức embedding, không OR/AND](#dec-029--m5-hybrid-soft-fusion-exp-006-fusion-m%E1%BB%A9c-embedding-kh%C3%B4ng-orand)
- [DEC-030 — Dừng train; chọn phương pháp cuối bằng val F1 + pipeline EXP-FINAL-001](#dec-030--dung-train-chon-phuong-phap-cuoi-bang-val-f1--pipeline-exp-final-001)
- [DEC-031 — Demo và báo cáo dùng bản 512 cho M1/M2/M5 (thay bản 256)](#dec-031--demo-va-bao-cao-dung-ban-512-cho-m1m2m5-thay-ban-256)
- [DEC-032 — Bộ test mở rộng nhóm P5: danh sách liên kết + hướng đối tượng](#dec-032--bo-test-mo-rong-nhom-p5-danh-sach-lien-ket--huong-doi-tuong)
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
- Ước lượng phần còn lại (POJ-104) **tại thời điểm DEC-019**: EXP-003 ~7.8h,
  EXP-004 ~2h, EXP-005 ~2h, EXP-006 ~8h → ~20h. Số `~2h` của EXP-004/005 là
  ước lượng suông, **đã bị thu hồi ở DEC-021** (đo thật ra ~35 phút).
- Ảnh hưởng tài liệu: `docs/research_questions.md` (thêm ghi chú scope),
  `docs/project_overview.md`, `README.md`, `docs/experiment_registry.md`
  (EXP-007…012 → CANCELLED).

[🔝 Quay lại mục lục](#top)

---

## DEC-020 — M2 dùng chung code path với M1, không clone runner

- **Bối cảnh:** RQ1 hỏi "GraphCodeBERT có tốt hơn CodeBERT không?". M1 và M2 cùng
  là RoBERTa-base 125M, vocab 50265, cùng hyperparameter. Nếu EXP-003 chạy một
  bản sao riêng của runner thì bất kỳ sửa đổi nào ở một bên (early stopping,
  threshold, error analysis) sẽ lọt vào kết quả so sánh dưới dạng "GraphCodeBERT
  tốt/bad hơn" — tức là đo **implementation gap** thay vì **pretrained weights**.
- **Quyết định:** tách `transformer_pair_runner` (copy nguyên hành vi từ EXP-002
  rồi generic hóa đúng một class) và `TransformerPairClassifier`; `exp002_codebert`
  thành thin wrapper giữ nguyên public API cũ, `exp003_graphcodebert` dùng **cùng
  tập** re-export. `tests/test_exp003.py` khóa bất biến này bằng assertion
  `is` (cùng đối tượng function) chứ không phải bằng review tay.
- **Lý do không "clone + sửa cho khớp":** hai bản sẽ trôi dần theo thời gian và
  không có gì báo động. Ngay trong lần làm này, `exp003` đã lệch API so với
  `exp002` (chỉ export `main`) — test bắt được ngay.
- **Bằng chứng công bằng:** cùng kiến trúc ⇒ cùng số tham số
  (`126,416,641` cho cả hai) và cùng vocab ⇒ **tokenization giống hệt**, được
  assert bằng `torch.equal(input_ids)` trên cùng đoạn code.
- **Cái giữ nguyên:** `configs/exp002_codebert.yaml` và
  `configs/exp003_graphcodebert.yaml` phải khớp nhau ở mọi mục trừ `model.hf_id`
  và tên/output — cũng được test tự động.
- **Hệ quả:** không thể "chỉnh EXP-003 cho đỡ" mà không sửa test. Đó là chủ ý.

[🔝 Quay lại mục lục](#top)

---

## DEC-021 — M3 ASTNN: vectorize RvNN, cache cây AST, và **đo thay vì đoán** thời gian

- **Bối cảnh:** M3 là model khác họ (GRU 428k tham số, train từ scratch), nên
  **không** áp chung hyperparameter với M1/M2 (DEC-016 chỉ áp cho M1/M2). Vẫn
  giữ chung seed, số epoch, warmup ratio, early stopping và cách chọn threshold
  để so sánh ở mức protocol.
- **Ba quyết định kỹ thuật:**

  1. **Abstraction đúng bài gốc:** lá chỉ còn `ID`/`NUM`/`STR`; node trong giữ
     **loại statement**. Mỗi lá gắn với statement cha **gần nhất** — đúng nghĩa
     "tách raw AST thành rừng cây statement" của Zhang et al. 2019, không phải
     gán cho *mọi* statement tổ tiên.
  2. **RvNN vectorize hoàn toàn** (`VectorizedRvNN`): gộp cây của cả batch thành
     mảng phẳng, duyệt **theo tầng độ sâu**, mỗi tầng là 1 matmul + 1
     `index_add`. Bản gốc đệ quy từng node bằng Python; với POJ-104 (~55
     statement, ~232 lá/code, 332k pairs × 3 epoch) phần đệ quy đó chiếm
     **giờ CPU**, không phải giờ GPU.
  3. **Cache cây `.npz`** (`ASTTreeStore`): compile một lần thay vì mỗi epoch.
     ~25 giây cho 41.491 codes, 231 byte/code nén.

- **Về "chính xác" của thời gian chạy — bài học rút ra:** ước lượng `~2h` ban
  đầu là **suy đoán**, đã bị thu hồi. Cách làm thay thế:
  - `diagnostic` đo `sec_per_optimizer_step` trên **batch thật**, có warmup
    loại khỏi phép đo, `torch.cuda.synchronize()` trước khi đọc đồng hồ, và mẫu
    pair **ngẫu nhiên có seed** (lấy "N cái đầu tiên" sẽ thiên lệch về code đơn
    giản → lạc quan).
  - M3 chọn `batch_size 32 × accum 1` và M1/M2 là `16 × 2` — cùng **31.125
    optimizer step**, nên so sánh 1-1.
  - Máy local là RTX 5060, Kaggle là T4. Để không báo cáo nhầm, đo hiệu chuẩn bằng
    chính EXP-002: local `0,4716 s/step` (spread 0,467–0,481) so với T4 thật
    `0,8985 s/step` ⇒ **T4 chậm hơn local 1,90×**. Áp hệ số đó cho M3
    (`0,0352 s/step` local) ⇒ **~34,8 phút**, nhanh hơn EXP-002 **13,4×**.
  - Con số này vẫn là **ước lượng**, vì hệ số 1,90× đo trên workload transformer
    chứ không phải GRU. Bắt buộc chạy lại `--mode diagnostic` trên Kaggle T4 để
    chốt trước khi đưa vào báo cáo.

- **Ba lỗi im lặng đã bắt được — đều không lộ ra qua smoke/diagnostic:**
  1. `nn.GRU` với input `PackedSequence` trả **2-tuple** `(output, h_n)`, khác
     LSTM là `(output, (h_n, h_c))`. Unpack 3 tầng cắt mất chiều batch.
  2. `VectorizedRvNN` trả `h_acc[:n_stmt]` — tức **tổng h của các con** chứ không
     phải `h(v)`, khiến **mọi node lá bằng 0**. Model vẫn chạy, loss vẫn giảm,
     gradient vẫn hữu ích ⇒ mọi test finiteness đều xanh. Chỉ khi so với **bản
     đệ quy tham chiếu viết độc lập** mới lộ (nay khớp tới `1,3e-06`).
  3. Đường dẫn cache AST không tách smoke với train đầy đủ ⇒ smoke (vài trăm
     code) ghi đè cache train (41k code), lần train sau **âm thầm chạy trên vài
     trăm code**, không cảnh báo nào. Nay cache path tách theo `tag` và
     `max_code_chars`.

  Kết luận chung: **test finiteness không chứng minh mô hình đúng.** Bản tham chiếu
  độc lập mới làm được. Đây là lý do `tests/test_exp004.py` có
  `_reference_rvnn` viết lại từ công thức bài báo thay vì gọi lại code production.
- **Phương pháp:** ASTNN **không truncate** (giữ toàn bộ cây statement) nên không
  chịu giới hạn `max_length 256` mà M1/M2 đang cắt mất 60,5% token. Đây là
  điểm cần nêu trong discussion — thuận lợi của M3 *không* phải do M3 "thông
  minh hơn" mà do cách biểu diễn, nên cũng là điểm cần thận trọng khi so sánh.
- **Ước lượng còn lại (POJ-104):** EXP-003 ~7,8h (đo), EXP-004 ~0,6h (đo),
  EXP-005 **chưa đo** (cần Joern theo DEC-009), EXP-006 ~8h.

## DEC-022 — Đóng băng recipe M1/M2 sau run EXP-002; evaluate chạy local

- **Bối cảnh:** EXP-002 chốt test **F1 0.910** (`reports/011_codebert_results.md`) và lộ ra
  ba vấn đề đo được: (1) `max_length 256` cắt **78,6 %** code train, tỉ lệ lỗi tăng từ
  5,2 % (quartile ngắn) lên 12,4 % (quartile dài); (2) `early_stopping` **không có
  `min_delta`** nên `patience` không bao giờ ăn — epoch 3 lãng phí ~2,3 h; (3) overfit biểu
  hiện ở `val_loss` (0.377 → 0.442) nhưng **khác** ở ranking (`val_roc_auc` 0.976 → 0.977).
- **Quyết định 1 — không sửa recipe giữa chừng:** `configs/exp003_graphcodebert.yaml`
  giữ nguyên `max_length: 256` và **không** thêm `min_delta`, đúng bằng EXP-002. Cải thiện
  (`max_length 512`, `min_delta: 0.005`) dồn thành **ablation riêng sau khi có đủ M1/M2**,
  không sửa lên artifact của run đã xong — `config_sha256` trong `run_metadata.json` phải
  còn khớp với config trong repo.
- **Quyết định 2 — evaluate là job rẻ, chạy local:** `--mode evaluate` chỉ cần
  `best_checkpoint.pt` + dataset và mất **5 m 16 s** trên RTX 5060, so với việc mở phiên
  Kaggle (giới hạn 30 h/tuần). Từ giờ chỉ `--mode train` mới cần Kaggle. Checkpoint không
  commit (`.gitignore` loại `*.pt`) nên phải giữ file tải về từ Kaggle cho tới khi hết
  nhu cầu resume.
- **Quyết định 2b — tải checkpoint BẢN NHẸ, không tải checkpoint đầy đủ (2026-09-28):**
  `best_checkpoint.pt` = 1,44GB vì nhồi optimizer + scheduler + NumPy RNG state, trong khi
  evaluate chỉ dùng `model_state`. Thêm `export_eval_weights()` xuất
  `best_weights_only.pt` (**0,51GB**, chỉ `model_state`/`epoch`/`best_f1`, load được với
  `weights_only=True`) và `resolve_eval_checkpoint()` để `evaluate_model` ưu tiên
  checkpoint đầy đủ, không thì fallback bản nhẹ. Kết quả: tải về **0,5GB thay vì
  1,44GB** (2,9GB nếu lấy cả `best` + `last`) mà metric **không đổi** — đã kiểm chứng
  trên chính artifact EXP-002 (F1 `0.9099191224319048`, TP/TN/FP/FN giống hệt). Notebook
  Kaggle đổi từ `tar --exclude='*.pt'` (loại luôn bản nhẹ ⇒ không còn gì để evaluate) sang
  `--exclude='*checkpoint.pt'`. Nguyên tắc: **mọi job chạy lại được từ artifact nhẹ hơn
  phải ưu tiên artifact nhẹ hơn**, kể cả khi file đầy đủ có "đủ thông tin hơn".
- **Quyết định 3 — sửa 3 lỗi im lặng, không đụng số liệu:** (a) `torch.load` thiếu
  `weights_only=False` khiến **mọi** `evaluate` chết sau khi train xong (torch ≥ 2.6 đổi
  mặc định, checkpoint có `np.random.get_state()`); (b) `exp002_codebert` /
  `exp003_graphcodebert` thiếu `if __name__ == "__main__": main()` sau khi refactor
  DEC-020 ⇒ `python -m` **exit 0 với 0 output**, tức EXP-003 sẽ "chạy 8 h không in gì";
  (c) `eval_time_s` khởi tạo timer sai chỗ nên chỉ đo thời gian ghi file. Cả ba đều là
  lỗi im lặng — đúng bài học DEC-021: test finiteness không chứng minh đúng.
- **Hệ quả:** số liệu EXP-002 giữ nguyên; EXP-003 chạy đúng recipe nên so sánh M1 vs M2
  hợp lệ; nhưng **kết luận về truncation vẫn là limitation mở**, phải nêu trong discussion
  thay vì giấu bằng cách đổi config.

[🔝 Quay lại mục lục](#top)

---

## DEC-023 — Notebook riêng cho M3 và M2, chạy M3 trước; evaluate ở Kaggle; EXP-005 là blocker

- **Bối cảnh (2026-09-28):** EXP-002 đã xong. EXP-003 (M2, ~7,8h) và EXP-004 (M3, ~35 phút)
  cùng `READY_FOR_KAGGLE_TRAIN`. Người dùng muốn train một lần cho cả hai, rồi yêu cầu
  **tách thành 2 notebook riêng**, **mọi notebook phải có cell evaluate chạy trên Kaggle**
  (không tải checkpoint 1,44GB về chạy lại local), và hỏi M4 (CPG-GNN) là gì.
- **Quyết định 1 — 2 notebook riêng, chạy M3 TRƯỚC M2:** một notebook cho mỗi mô hình
  (`notebooks/exp004_astnn_kaggle.ipynb`, `notebooks/exp003_graphcodebert_kaggle.ipynb`),
  mỗi cái tự build dataset và tự đóng gói. Bản gộp một phiên đã bị bỏ. Thứ tự chạy vẫn
  là **M3 trước M2** vì **phụ thuộc, không phải thẩm mỹ**: M3 rẻ hơn 13×, nếu phiên M2
  chết giữa chừng thì đã có M3. Cả hai nằm trong một ngày quota 30h GPU của Kaggle
  (~8,4h < giới hạn 12h/phiên).
- **Quyết định 2 — evaluate chạy trên Kaggle, tải về chỉ metrics + predictions:**
  evaluate tốn **5 phút** (M2) / **~1 phút** (M3) trên GPU Kaggle, còn tải
  `best_checkpoint.pt` (1,44GB) hoặc `best_weights_only.pt` (0,5GB) về máy để chạy lại
  thì tốn thời gian tải + giờ GPU local, thuộc loại việc **tính lại được từ artifact**.
  Vì `predictions/{validation,test}_scores.jsonl` đủ để tính lại mọi metric, threshold
  và error analysis, notebook mặc định **không** tải checkpoint về
  (`KEEP_WEIGHTS = False`); bật cờ mới giữ `best_weights_only.pt`. Đây là sửa đổi so với
  quyết định 2 của DEC-022 (lúc đó chọn evaluate local vì sợ mất quota GPU) — cùng một
  nguyên tắc đằng sau cả hai: **ưu tiên cách rẻ nhất để chạy lại**, và quota 30h/tuần
  không phải thứ nên tiêu cho 6 phút.
- **Quyết định 3 — EXP-005 không làm trong phạm vi hiện tại:** ba lý do độc lập, đều đo
  được — (a) `joern` **chưa cài** và máy **không có Java** (DEC-009 để ngỏ; kiểm lại
  2026-09-28 vẫn `not recognized`); (b) tree-sitter đo được **3.153 file/s** trên POJ-104
  (500 file thật, 0,16s) trong khi Joern cần JVM + front-end C/C++ đầy đủ, chậm hơn
  1–2 đơn vị ⇒ CPG cho ~62k code tốn **1,4–17 giờ**; (c) chưa có model CPG-GNN —
  `src/representations/cpg/build.py` mới chỉ là stub `cpg_status()`, chưa có config,
  chưa có training loop, chưa có test. Kaggle vẫn train GPU bình thường (T4, 30h/tuần) —
  vấn đề của M4 **không phải GPU**, mà là toolchain + thời gian dựng CPG.
- **Việc còn lại:** EXP-005 là milestone riêng — cài JRE 17 + Joern, đo throughput thật
  trên 500 file, rồi mới quyết định có làm M4 hay không. Không sửa `PLANNED` thành vời;
  không thay M4 bằng model khác (DEC-009).

[🔝 Quay lại mục lục](#top)

---

## DEC-024 — M4 contract-first: đo thay vì cài, Joern chạy trên Kaggle không cài local

- **Bối cảnh (2026-09-28, sau DEC-023):** người dùng hỏi "có cần test ở local không,
  train ở Kaggle thì khỏi cài". Câu trả lời là **không** — và nguyên nhân kỹ thuật
  cũng là lý do để quyết định này: phần bị chặn không phải code M4 mà là **một lệnh
  cài**. Nên tách việc: viết và test mọi tầng sau trích xuất (không cần Joern), còn
  Joern thì chạy ở nơi duy nhất mà train cần — Kaggle.
- **Quyết định 1 — contract-first, không chờ Joern:** vocabulary CPG được **đóng băng**
  trước trong `src/representations/cpg/schema.py` (13 nhãn node, 9 nhãn cạnh), và mọi
  tầng phía sau — `ingest` (JSON → `CPGraph`), `CPGStore` (cache `.npz`), `CPGGNN` (GAT
  có điều kiện nhãn cạnh, 204.801 tham số) — được test bằng **đồ thị tổng hợp** đúng
  format mà `scripts/joern/export_cpg.py` sẽ ghi ra (33 test, xanh local, không cần
  Java/GPU). Vì sao hợp lệ: format JSON là **do ta định nghĩa**, không phải format nội
  bộ Joern; phần chưa kiểm chứng chỉ còn đúng một chỗ — exporter có ghi đúng format đó
  hay không. Nếu chờ Joern rồi code, ta sẽ không có gì để test cho tới khi lên Kaggle.
- **Quyết định 2 — JSON format tự định nghĩa:** exporter phía Joern chỉ giữ đúng các
  nhãn trong schema; nhãn lạ → `OTHER` (node) hoặc bỏ (cạnh), **luôn được đếm** trong
  stats. Lý do: Joern đổi tên node type giữa các version, và vocabulary đầu vào mô
  hình phải đóng băng — đổi vocabulary = đổi feature = mọi checkpoint cũ vô dùng.
- **Quyết định 3 — Joern cài trên Kaggle, KHÔNG cài local:** máy dự án không có Java và
  không cần có. Kaggle có sẵn JDK 17 + quota CPU, nên `notebooks/exp005_cpg_extract_kaggle.ipynb`
  cài rồi đo ngay trong phiên đó. Cài local tốn hàng trăm MB, đụng môi trường đang xanh,
  và không giúp ích gì cho lần train vì lần train vẫn ở Kaggle.
- **Quyết định 4 — số đo thay cho ước lượng, và đo trước khi cam kết:** notebook đo
  20 hàm rồi 500 hàm, ghi `artifacts/cpg_export/benchmark.json`
  (`functions_per_second`, `error_rate`, `est_hours_full_dataset`). Trước khi chạy 62k
  hàm (5–17 h) phải thấy con số đó. Ngưỡng: **< 0,3 hàm/s ⇒ 62k hàm > 1 ngày ⇒ phải
  bàn lại phạm vi** (giảm số hàm, hoặc song hoá 4 tiến trình) thay vì đốt quota.
- **Quyết định 5 — gate chất lượng CPG sau khi nạp:** `nodes_per_code_mean < 20` ⇒
  exporter hỏng, dừng; `edges_dropped_*` lớn ⇒ lệch schema, sửa trước khi train;
  thiếu nhãn `REACHING_DEF` ⇒ đồ thị không còn là CPG, dừng. Ba gate này rẻ hơn nhiều
  so với một lần train 5 h ra số rác.
- **Hệ quả:** `configs/exp005_cpg_gnn.yaml` mang `status: BLOCKED_ON_JOERN` +
  `blocked_by: [java, joern]`, và `src/representations/cpg/build.py` báo cáo
  `ready: false` kèm hướng dẫn. Trạng thái này là **trung thực**: phần đã xong được
  ghi ra, phần chưa được ghi ra, không có gì được ghi là sẵn sàng.
- **Còn lại (không làm trong DEC này):** runner `src/experiments/exp005_cpg_gnn.py` và
  notebook train M4 — cố ý chờ số đo, vì chúng chỉ khác EXP-004 ở chỗ load CPG store;
  viết trước rồi mới sửa batch size theo số node thật là việc làm hai lần.

[🔝 Quay lại mục lục](#top)

---

## DEC-025 — Joern v4: script Scala + gom cả lô vào một tiến trình, và smoke test bắt buộc

- **Bối cảnh (2026-09-28, ngay sau DEC-024):** trước khi đóng gói EXP-005, phần
  Joern được đối chiếu với tài liệu chính thức — và **cả ba giả định ban đầu đều
  sai**. Sai thì tốn một lần chạy Kaggle dài, nên kiểm trước khi cài gì:
  1. **Ngôn ngữ lập trình:** Joern hiện là **v4.0.x** (phát hành bản mới hằng
     ngày), không phải v2.x. Asset tải về tên `joern-cli-linux-x86_64.zip` (~2GB),
     **không có** `joern.zip` như đoán trước đó ⇒ URL hardcode kiểu đó là công thức
     404.
  2. **Giao diện script:** Python (`from joern import cpgFromCode`) là của
     `joernpy` ở bản cũ. Bản v4 chạy script bằng **Scala** qua CLI
     `joern --script x.sc --param k=v` (đã đối chiếu docs.joern.io/interpreter).
  3. **Chi phí khởi động:** mỗi lần gọi `joern` là một JVM + scala-repl, tốn
     **vài chục giây**; dựng CPG một hàm thì dưới 1 giây.
- **Quyết định 1 — không hardcode version/tên asset:** dùng đúng cách cài chính
  thức (`joern-install.sh` tự dò platform, đặt binary ở `~/bin/joern`), và
  `build_cpg_export.py` tìm Joern theo thứ tự `$JOERN_BIN` → `PATH` →
  `~/bin/joern`. Lý do: Joern ra bản mới hằng ngày; bất cứ thứ gì ghim version
  trong repo sẽ hỏng trong vài tuần mà không ai sửa kịp.
- **Quyết định 2 — exporter là Scala, và mỗi loại cạnh một hàm riêng:** không có
  lựa chọn nào khác cho Joern v4, nhưng để giữ rủi ro thấp khi **không kiểm chứng
  được**: `astEdges`/`callEdges`/`argEdges`/`refEdges`/`reachingDefEdges`/`cfgEdges`
  tách bạch, mỗi cái một hàm — sửa một tên hàm không phá các chỗ khác.
- **Quyết định 3 — bắt buộc có smoke test trước lô lớn:** vì không có Java ở máy
  dự án, `scripts/joern/joern_smoke.sc` chạy trên **một** hàm và in ra danh sách
  step CPGQL mà chính Joern khai báo (`cpg.help`) cùng các dòng probe độc lập
  (`ast=`, `callOut=`, `reachingDef=`…). Nguyên tắc: **phép đo rẻ đi trước lỗi
  đắt đi sau**; một lần chạy 60 giây thay cho một lần chạy 500 hàm rồi mới biết
  mình sai. Cell smoke được đặt ngay sau cell cài Joern và được ghi rõ là
  **không được bỏ qua**.
- **Quyết định 4 — một tiến trình cho cả lô, không phải một tiến trình mỗi hàm:**
  `--batch-size` file được ghi thành `NNNNN.cpp` trong một thư mục, kèm
  `index.tsv` ánh xạ tên file → `sample_id`, rồi Joern được gọi **một lần cho cả
  lô**. Con số này quyết định toàn bộ ngân sách thời gian: 62.000 tiến trình × 20s
  khởi động ≈ **15 ngày** chỉ để bật JVM, trong khi gom lô 100 hàm/lần thì chi phí
  khởi động là 620 lần khởi động cho cả dataset. Bắt buộc phải có
  `index.tsv` vì Joern chỉ trả về *tên file*, còn `sample_id` của dataset có dạng
  `poj104-train-7` — suy ra không được.
- **Quyết định 5 — `.meta` cạnh mỗi CPG:** mỗi hàm xuất kèm
  `n_methods_in_file`, `n_nodes`, `n_edges`, `method_name`. Bốn số này là gate
  chất lượng: POJ-104 mỗi dòng là một hàm, nên `n_methods_in_file > 1` là dấu
  hiệu ta đang lấy **sai hàm** — thứ mà nhìn bằng mắt trên 500 file sẽ bỏ sót.
- **Hệ quả:** `export_cpg.py` bị xoá (đường `joernpy` không còn là đường chính
  thức); `benchmark.json` nay có thêm `n_missing`/`nodes_per_cpg_mean`/
  `files_with_multiple_methods`, và **hàm không ra JSON cũng bị tính là lỗi**
  (`error_rate` gộp cả hai) — trước đó một lô timeout sẽ bị báo cáo là "0 lỗi".
- **Còn lại:** tên step CPGQL trong `export_cpg.sc` chỉ được xác nhận sau khi cell
  smoke chạy trên Kaggle. Không coi EXP-005 là "đã xong" cho tới lúc đó.

[🔝 Quay lại mục lục](#top)

---

## DEC-032 — Bộ test mở rộng nhóm P5: danh sách liên kết + hướng đối tượng
- **Bối cảnh (2026-10-17):** thầy góp ý bộ test `custom_suite` "khá ngắn": P1–P4 chỉ 11–17
  dòng, chưa có hàm, chưa có cấu trúc dữ liệu, chưa có hướng đối tượng, và chưa có cặp nào
  gần như hoàn toàn giống nhau giữa `struct` và `class`. Bộ test cũ vì thế chưa kiểm tra
  đúng loại biến đổi hay gặp nhất ở bài nộp (bọc `struct` + hàm tự do thành `class` +
  phương thức).
- **Quyết định 1 — thêm nhóm P5, 6 file C++ dài 59–71 dòng** vào `data/custom_suite/`:
  `p5_base_list.cpp` là bản gốc (`struct Node` + 4 hàm tự do `pushBack/sumList/printList/freeList`,
  đọc $n$ số, in dãy rồi in tổng); `p5_v_class.cpp` là **cùng hành vi** nhưng viết bằng
  `class IntList` + 4 phương thức, con trỏ `head` để trong đối tượng; ba biến thể clone
  `p5_v_while` (for→while + đổi tên biến), `p5_v_scanf` (cin/cout→scanf/printf + đổi tên),
  `p5_v_combo` (while + scanf + đảo điều kiện + biến thừa); `p5_n_palindrome.cpp` dùng **cùng
  khung** `class` + `Node` nhưng khác ngữ nghĩa (kiểm tra dãy đối xứng) làm non-clone khó.
  Không file nào có comment; giữ `#include <bits/stdc++.h>` + `using namespace std;`, riêng
  hai bản `scanf` dùng `#include <cstdio>`.
- **Quyết định 2 — 8 cặp mới trong `pairs.jsonl` (26 → 34 cặp):** `P5_struct_class`
  (struct ↔ class), `P5_while`, `P5_scanf`, `P5_combo`, `P5_class_combo`,
  `P5_nonclone_palindrome`, và hai cặp cross `X_P1P5` (DSLK vs tổng 1..n), `X_P2P5`
  (DSLK vs UCLN).
- **Kiểm chứng code trước khi chấm:** 6/6 file biên dịch sạch bằng
  `g++ -std=c++17 -O2 -Wall`; 5 bản tính tổng cho **cùng kết quả** trên cùng input
  (`1 2 3 4 5` → `1 2 3 4 5` và `15`); `p5_n_palindrome` trả `NO`/`YES` đúng. Mọi dòng ≤ 46
  ký tự (đo được: minipage 0,49\textwidth chứa ≤ 49 ký tự ở `\footnotesize`, ≤ 54 ở
  `\scriptsize`) để hình hai cột không bị xuống dòng.
- **Quyết định 3 — đưa `data/custom_suite` vào git.** `.gitignore` đổi `data/` → `data/*` +
  `!data/custom_suite/` (git **không** re-include được file khi cả thư mục cha bị loại).
  Dữ liệu lớn (`data/pairs`, `data/processed`, `data/splits`) vẫn bị ignore. Bộ test là code
  tự viết (~10 KB) và là nguồn của hình trong báo cáo nên phải nằm trong repo.
- **Công cụ:** `scripts/score_custom_suite.py` thêm `--pairs` (chấm riêng cặp mới, dùng chung
  `--merge`); `scripts/evidence_custom_suite.py` thêm `--models` (chỉ vẽ hình cho model được
  chọn, JSON vẫn đủ 5 model) và `fig_font` tự dùng `\scriptsize` khi file dài (> 48 dòng);
  `scan_demo_zip.py::_latex_minipage` thêm tham số `font`. Báo cáo mới đặt
  `\setlength{\fboxsep}{1pt}` vì `\colorbox` mặc định cộng 3pt mỗi phía làm mỗi dòng tô màu
  cao thêm ~6pt (71 dòng là tràn trang).
- **Kết quả (6 phương pháp, ngưỡng đọc từ `threshold.json` của từng run):**

  | Cặp | GT | B1 MOSS (0,010) | M1 (0,210) | M2 (0,660) | M3 (0,420) | M4 (0,470) | M5 (0,010) |
  | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
  | P5_struct_class | clone | **0,000** | 1,000 | 1,000 | 0,915 | 0,973 | 1,000 |
  | P5_while | clone | 0,540 | 1,000 | 1,000 | 0,993 | 0,973 | 1,000 |
  | P5_scanf | clone | 0,710 | 1,000 | 1,000 | 0,995 | 0,973 | 1,000 |
  | P5_combo | clone | 0,150 | 1,000 | 1,000 | 0,994 | 0,973 | 1,000 |
  | P5_class_combo | clone | **0,000** | 1,000 | 1,000 | 0,907 | 0,973 | 1,000 |
  | P5_nonclone_palindrome | non-clone | 0,000 | 0,002 | 0,000 | 0,270 | **0,743** | 0,002 |
  | X_P1P5 | non-clone | 0,000 | **1,000** | **1,000** | 0,179 | **0,742** | **1,000** |
  | X_P2P5 | non-clone | 0,000 | 0,000 | 0,000 | 0,375 | **0,710** | 0,000 |

  - **Cặp thầy quan tâm** (`P5_struct_class`, `P5_class_combo`): cả 5 model đều nói đúng
    clone (M1/M2/M5 ≈ 1,000; M3 0,915 / 0,907; M4 0,973), còn **MOSS bỏ sót cả hai** (0,000)
    — đúng hành vi đã thấy ở bộ test cũ (server chỉ gắn cờ khi có chùm fingerprint dài).
  - Non-clone khó `P5_nonclone_palindrome`: chỉ **M4** sai (0,743 > 0,470); M3 0,270 và
    M1/M2/M5 0,000–0,002 đều đúng ⇒ **đảo chiều** so với bộ test cũ, nơi M3 và M4 cùng bị
    khung code lừa (10 cặp mỗi model).
  - **Lỗi mới ở cặp cross:** `X_P1P5` bị M1/M2/M5 ≈ 1,000 và M4 0,742 nói sai là clone, chỉ
    M3 đúng (0,179); ở 6 cặp cross cũ M1/M2/M5 đều ≤ 0,174. Hai bài **cùng phép toán tổng**
    và cùng khung `main` (đọc $n$, vòng `for`, `cout`, `return`) nên cặp khó hơn hẳn về ngữ
    nghĩa, không chỉ về hình thức. `X_P2P5` chỉ M4 sai.
  - Tổng 8 cặp sai: B1 2, M1 1, M2 1, M3 **0**, M4 3, M5 1. Trên cả 34 cặp: B1 18, M1 3,
    M2 3, M3 10, M4 13, M5 4 lỗi (M3/M4 trước đây mỗi model 10 lỗi trên 26 cặp cũ).
- **Giới hạn vừa lộ ra:** cả 6 file P5 vượt 512 subword token (520 ở `p5_v_scanf` tới 683 ở
  `p5_v_class`, đo bằng tokenizer GraphCodeBERT), nên M1/M2/M5 (`max_length: 512`) chỉ đọc
  phần đầu: cắt 1,7 % ở `p5_base_list` (521 token) và tới **25 %** ở `p5_v_class` (683 token).
  M3 (cây AST) và M4 (CPG) không bị cắt — CPG của 6 file chỉ 61–95 node, xa trần
  `max_nodes: 1500`. Muốn test code dài hơn nữa thì phải nới `max_length` hoặc cắt theo hàm
  (việc đã hoãn từ DEC-029).
- **Hệ quả:** báo cáo tuần mới `reports/[24 - 10] - Báo cáo.tex` (6 trang, PDF kèm theo) chỉ
  nói phần này: 6 file + 8 cặp, trích code struct↔class hai cột, bảng điểm 8 cặp × 6 phương
  pháp, hai hình minh hoạ M2 (`P5_struct_class` đúng, `X_P1P5` sai). `reports/[17 - 10] -
  Báo cáo.tex` **không đổi** (không động vào số cũ).

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
| 12 | 1.11.0 | 2026-09-27 | Thêm DEC-020 (M2 dùng chung runner với M1, khóa bằng test `is`), DEC-021 (M3 ASTNN xong: vectorize RvNN, cache AST, **đo thay vì đoán** — ước lượng ~2h bị thu hồi còn ~35 phút). |
| 13 | 1.12.0 | 2026-09-28 | Thêm DEC-022 sau khi EXP-002 có số liệu: đóng băng `max_length 256` + chưa thêm `min_delta` cho M1/M2 (đổi recipe giữa chừng sẽ phá so sánh RQ1), evaluate chạy local 5 phút thay vì mở phiên Kaggle, sửa 3 lỗi im lặng (`weights_only`, mất `__main__` guard, `eval_time_s` đo sai). |
| 14 | 1.13.0 | 2026-09-28 | DEC-022 bổ sung quyết định 2b: xuất `best_weights_only.pt` (0,51GB) để tải về chạy evaluate thay vì tải checkpoint 1,44GB, kèm `resolve_eval_checkpoint` fallback; sửa tar trong notebook EXP-002/003 vốn loại luôn bản nhẹ. |
| 15 | 1.14.0 | 2026-09-28 | Thêm DEC-023: **2 notebook riêng** cho M3 và M2 (bản gộp 1 phiên bị bỏ), chạy M3 trước M2 (phụ thuộc, không phải thẩm mỹ); **evaluate chạy trên Kaggle** và chỉ tải metrics + predictions về (sửa hướng của DEC-022 §2); chốt EXP-005 là blocker (không có Joern/Java, CPG ~62k file, chưa có model — vấn đề là toolchain chứ không phải GPU). |
| 16 | 1.15.0 | 2026-09-28 | Thêm DEC-024: M4 làm **contract-first** (schema → ingest → store → GAT, 33 test local, không cần Joern); JSON format tự định nghĩa để đổi backend không phá model; **Joern cài trên Kaggle, không cài local**; số đo `ham/giay` + gate chất lượng CPG trước khi cam kết 5–17 h. |
| 17 | 1.16.0 | 2026-09-28 | Thêm DEC-025: đối chiếu tài liệu Joern → **v4.0.x, script Scala** (`joern --script x.sc --param k=v`), **không hardcode version/tên asset** (`joern-install.sh`); exporter `.py` bị xoá; **gom lô vào 1 tiến trình** (62k tiến trình × JVM ≈ 15 ngày); **smoke test bắt buộc** trước lô lớn; `error_rate` nay gộp cả hàm không ra JSON. |
| 18 | 1.17.0 | 2026-10-03 | Thêm DEC-026: **baseline báo cáo là MOSS (EXP-013), không dùng TF-IDF** — EXP-001 chỉ còn vai trò tham khảo. Tham số gốc MOSS (`k=50,w=100`) **không dùng được** trên POJ-104 (hàm quá ngắn ⇒ mỗi hàm chỉ ~6 fingerprint ⇒ F1 rơi về 0,667 = dự đoán toàn bộ là dương); dò trên validation, chọn `k=20,w=5`. Ghi nhớ tồn đọng: `moss.py` còn 5 test fail, bug `normalize` nuốt dấu `=` ở `*=` không có khoảng trắng. |

---

## DEC-026 — Baseline của báo cáo là MOSS (EXP-013), không dùng TF-IDF
- **Bối cảnh (2026-10-03):** trước đây EXP-001 (TF-IDF) được coi là baseline vì
  đã có sẵn và đã qua test. Vấn đề là TF-IDF là mô hình bag-of-token của **chính
  nhóm**, nên khi dùng làm gốc so sánh rồi kết luận về mô hình học sâu thì bất
  công: thắng ở đây chưa chắc thắng ở đó. Và TF-IDF không hề bịt cả thông tin về
  cấu trúc hay dữ liệu, nên nó không phải đối thủ xứng tầm.
- **Quyết định 1 — MOSS (EXP-013) là baseline chính, dùng cho mọi báo cáo.**
  MOSS là thuật toán đã có tuổi và được chấp nhận rộng rãi (Schleimer, Wilkerson &
  Aiken 2003), chạy lặp lại được không cần mạng, nên đây là đối thủ công bằng. **Không
  dùng TF-IDF làm baseline.** EXP-001 vẫn giữ trong registry ở trạng thái
  `COMPLETED (tham khảo)` — số liệu còn nguyên, chỉ là không dùng làm gốc.
- **Quyết định 2 — không dùng tham số gốc của MOSS.** Bài báo gốc dùng
  `k=50, w=100`, chọn cho *văn bản dài*. Hàm POJ-104 sau chuẩn hoá dài trung vị
  chỉ 349 ký tự, nên cửa sổ winnow rộng `100 × 50 = 5.000` ký tự là **rộng hơn cả
  hàm**: đo được mỗi hàm chỉ còn ~6 fingerprint, similarity bằng 0 với mọi cặp, và
  F1 rơi đúng 0,667 — bằng đúng F1 của "dự đoán toàn bộ là dương". Trong lưới
  12 tổ hợp, 9 tổ hợp rơi vào đúng mốc 0,667 này. Đã dò `(k, w)` **trên
  validation** và chọn `k=20, w=5` (F1 validation 0,729), đóng băng rồi mới chạy
  test một lần. Test: P 0,646 / R 0,722 / F1 0,682 / ROC-AUC 0,731.
- **Quyết định 3 — `scripts/eval_moss.py` là runner chính thức của EXP-013.**
  Chạy CPU vài phút, không cần Kaggle. Threshold ghi kèm `k`, `w` trong
  `threshold.json` để kết quả luôn đi kèm cấu hình đã dùng.
- **Đính chính (2026-10-03, ghi đè đoạn "tồn đọng" ở bản 1.17.0 đầu):** nhóm
  nghi ngờ `normalize` nuốt dấu `=` ở `*=` không khoảng trắng, và từ đó suy ra
  "17% hàm / 30% cặp bị ảnh hưởng". **Kết luận đó sai.** Đã kiểm chứng:
  1. `a*=x` → `id*=id;` — không mất `=`. Đúng cả khi có lẫn không khoảng trắng.
  2. Đếm trên toàn bộ 51.991 hàm POJ-104: `*=` 467 → 464, `/=` 652 → 513,
     `%=` 193 → 187, `>>=` giữ nguyên, `<<=` 1 → 0. Phần mất đều nằm trong
     comment hoặc chuỗi literal, tức là **đúng hành vi**. Không toán tử nào bị
     nuốt ở mã thực thi.
  3. `=` giảm 7.506 và `.` giảm 25.193 là do chuỗi literal bị quy về `str` —
     `printf("a.b?c,d")` mất hết dấu bên trong. Cũng đúng thiết kế.
  Vậy `src/models/baseline/moss.py` **không có bug**, và số liệu EXP-013 không
  bị ảnh hưởng: chạy lại toàn bộ (kể cả dò 12 tổ hợp) cho `metrics.json` **giống
  hệt** bản trước. 5 test fail là **kỳ vọng sai trong test**, đã sửa:
  - `normalize('const char *s = "hello";')` mong `idid*=id=str;` — trong mã nguồn
    không có toán tử `*=` nào, chỉ là khai báo con trỏ `char *s`. Đúng phải là
    `idid*id=str;`.
  - `normalize("a = 'x';")` mong `id=id=str;` — `'x'` là char literal, chỉ thành
    `str`, không thể vừa là `id` vừa là `str`. Đúng phải là `id=str;`.
  - 3 test còn lại dùng `k` mặc định 50 trên đoạn mẫu dài 35 ký tự nên ra 0
    k-gram. Đây là hành vi đúng; đã thêm `TOY_K, TOY_W = 5, 2` cho các test đó
    và thêm một test ghi rõ `k=50` trên code ngắn cho 0 k-gram.
  Bổ sung: `similarity` là **containment** (`|A∩B| / min(|A|,|B|)`), nên hàm ngắn
  rất dễ "nằm trong" hàm dài — bubble sort với hàm cộng vẫn ra ~0,81. Đây là lý
  do `eval_moss.py` **phải dò ngưỡng trên validation** thay vì dùng hằng số, và là
  phần giải thích sâu hơn cho việc MOSS đoán nhầm nhiều trên POJ-104.
- **Bài học rút ra (nhớ kỹ):** khi báo cáo chưa có sẵn một baseline công bằng thì
  **tự dựng và tự chạy**, đừng lấy số của mô hình yếu hơn cho tiện rồi kết luận
  sai. Số phải kèm được câu lệnh đã chạy, thời gian chạy, và khả năng tái lập.
  Và **đừng tin chẩn đoán của chính mình**: cả "có bug" lẫn "17% hàm bị ảnh
  hưởng" đều là suy đoán chưa kiểm, chỉ đến khi đếm trên dữ liệu thật mới thấy
  sai. Test fail ≠ có bug; phải hỏi test sai hay code sai trước khi sửa code.
| 19 | 1.18.0 | 2026-10-03 | **DEC-026 đính chính:** suy đoán “normalize nuốt dấu `=` ở `*=`” và “17% hàm / 30% cặp bị ảnh hưởng” đều **sai**. Đếm kiểm: `*=` 467→464, `+=` 17.882→17.876, `>>=` giữ nguyên; phần mất đều nằm trong comment hoặc chuỗi literal. `moss.py` **không có bug**, `metrics.json` chạy lại **giống hệt**. 5 test fail là kỳ vọng sai → đã sửa test (`idid*id=str;`, `id=str;`, thêm `TOY_K/TOY_W`). Phát hiện thêm: `similarity` là **containment** nên hàm ngắn dễ dễ “chứa” hàm dài (bubble sort vs hàm cộng ≈ 0,81) — lý do phải dò ngưỡng trên validation. |
| 20 | 1.19.0 | 2026-10-03 | Thêm DEC-027: **M4 schema phương án (b)** — bỏ CALL/REF/REACHING_DEF khỏi `EDGE_LABELS` (9→6 loại, giữ thứ tự tương đối); sửa PARAMETER_OUT bằng khớp tên (identity không bao giờ trúng, đo 0/500). |

---

## DEC-027 — M4 schema (b): bỏ CALL/REF/REACHING_DEF, sửa PARAMETER_OUT bằng khớp tên

- **Quyết định (người chốt, 2026-10-03):** phương án (b) — đổi hợp đồng schema M4 thay vì chạy dataflow frontend.
- **Căn cứ đo (EXP-005 cell 7/8/9, Kaggle):** 500/500 OK, error 0; 4,267 hàm/s → est 3,38 giờ cho 51.991 hàm (crosscheck 3,39, khớp); nodes/code 354,6, edges/code 1.188,4; 5 loại cạnh ổn định (ARGUMENT, AST, CFG, EVAL_ORDER, TARG); `truncated=0`, `edges_dropped_*=0`; synthetic fallback 8/500 (1,6%).
- **Căn cứ API (cell 5, 53 candidate):** mọi đường đọc cạnh data-flow qua DSL đều FAIL — `call.reachingDef*`, `identifier/local.reachingDef*`, `cpg.graph.E*`, `call.ref`, `outE.source/target`. Chỉ `node.outE/inE` per-node hợp lệ nhưng chưa rõ trả về gì. `outE.labels` xác nhận đồ thị CÓ cạnh REACHING_DEF/REF, chỉ là DSL không mở được.
- **Hợp đồng mới (`EDGE_LABELS`, giữ thứ tự tương đối):** AST, CFG, ARGUMENT, TARG, PARAMETER_OUT, EVAL_ORDER. Chưa train checkpoint nào nên đổi index edge an toàn (`model.py` dùng `N_EDGE_TYPES` động, mọi chỗ khác suy từ `len`).
- **PARAMETER_OUT 0/500 là bug exporter, không phải dữ liệu:** `m.parameter` (METHOD_PARAMETER_IN khai báo) và argument (biểu thức) không bao giờ là cùng object nên `IdentityHashMap.containsKey` luôn false. Fix: khớp theo tên (`g(x)` tham chiếu tới tham số `x`) — cần cell 5 xác nhận `parameter.name`/`parameter.code` + `argument.code` trước (vòng 6, ~40s), không đoán.
- **Hệ quả:** M4 mất data-flow xuyên hàm; TARG (kết quả gọi hàm → biến đích, nội hàm) là tín hiệu data-flow duy nhất giữ được. Gần ASTNN hơn nhưng vẫn hơn ở CFG + ARGUMENT + TARG + EVAL_ORDER.
- **Chưa Save Version:** đúng theo notebook — schema và exporter chưa xong thì chưa Save.
- **Kiểm chứng fix (2026-10-03, cell 5 vòng 6 + chạy lại cell 7/8/9):** `parameter.name`, `parameter.code`, `argument.code` cả 3 OK. PARAMETER_OUT 0→67 cạnh trên 500 hàm (`never_produced` còn `[]`); `edges_total` tăng đúng +67, không đụng gì khác. Hiếm (`missing_ratio` 0,992) là do dữ liệu — chỉ 17 `METHOD_PARAMETER_IN` node/500 hàm (POJ-104 đọc stdin, ít hàm có tham số). Gate mới: `Gate AST+CFG: PASS` + `Gate PARAMETER_OUT: PASS`, `edge_types_missing: []`. Số chốt Save: 4,416 hàm/s; est 3,27 giờ; error 0; nodes/code 354,6; đủ 6 loại cạnh.
| 21 | 1.20.0 | 2026-10-03 | DEC-027 kiểm chứng xong: vòng 6 cả 3 OK; PARAMETER_OUT 0→67 (+67 đúng bằng tăng `edges_total`); gate mới PASS hết, `missing: []`. Số chốt Save: 4,416 hàm/s, est 3,27h, error 0, 354,6 nodes/code, đủ 6 loại cạnh. |
| 22 | 1.21.0 | 2026-10-04 | ASTNN train tiếp trần 20 epoch (early stopping patience 2), train từ đầu: dừng epoch 7, tốt nhất epoch 4 (val F1 0,853). Test P/R/F1 0,760/0,894/0,822±0,004, ROC-AUC 0,889, thr 0,42; ~40 phút T4. Lưu ý: schedule 20-epoch khác schedule cũ nên epoch đầu đã khác (0,789 vs 0,643) — chỉ so best, không so từng epoch. Báo cáo cập nhật hàng ASTNN, bỏ “chưa train đủ”. Đính chính số ở dòng 21: 354,6 nodes/code chỉ là 500 hàm train đầu (samples.jsonl xếp theo nhóm nên không đại diện); TB thật cả 51.991 hàm ~150 (train ~148, val ~188, test ~141). |
| 23 | 1.22.0 | 2026-10-04 | M4 train xong (5 epoch, ~57 phút T4): val F1 còn tăng tới epoch cuối (0,800→0,817→0,819) nên dừng vì hết epoch chứ không plateau — như ASTNN lúc trước. Test P/R/F1 0,695/0,885/0,779±0,004, ROC-AUC 0,864, thr 0,42. Kém CodeBERT 0,131 (CI [-0,140;-0,122], có ý nghĩa), hơn MOSS 0,097. Báo cáo thêm hàng CPG-GNN (5 phương pháp), bỏ “Tiếp tục CPG-GNN”. |
| 24 | 1.23.0 | 2026-10-07 | Thêm DEC-028: **baseline chính chuyển sang MOSS service thật (mosspy, EXP-014)** — thay thế quyết định 1 của DEC-026. EXP-013 (winnowing all-pairs local) hạ xuống `COMPLETED (tham khảo)`, giữ làm fallback demo (k=20,w=5). Đính chính citation: MOSS là **Schleimer, Wilkerson & Aiken 2003** (không phải Sleator & Bernstein). |
| 25 | 1.24.0 | 2026-10-07 | Thêm DEC-029: audit bổ trợ M2×M4 (test, 11.000 cặp) → M4 sửa 662/1.053 lỗi của M2 (62,9%) nhưng phá 2.377/9.947 đúng (23,9%), net −1.715 ⇒ **quy tắc OR/AND bị loại** (OR F1 0,809, AND F1 0,879, đều kém M2 0,907). **M5 = soft fusion mức embedding** (concat_mlp [hA;hB;|hA-hB|], DEC-005 + DEC-016): logistic (s_M2,s_M4) làm baseline đạt test F1 0,9217 > M2 0,9069 (+1,5đ) nhưng AUC 0,9683 < M2 0,9713 — thắng chỉ ở vùng ngưỡng ~0,535, phải nêu đúng vùng thay vì kết luận chung. Warm start encoder M2 (EXP-003) + M4 (EXP-005) rồi joint fine-tune LR 2e-5; giữ max_length 256 (so sánh sạch, không vội segment-GRU). Script: `scripts/audit_m2_m4_complementarity.py`, output `results/audit_m2_m4.json`. |
| 26 | 1.25.0 | 2026-10-07 | DEC-029 **§4 bổ sung — kênh phân phối checkpoint warm start = Hugging Face Hub.** Lý do: checkpoint `*.pt` bị `.gitignore` (M2 1,41GB) không thể trong repo; đồng thời khi chạy notebook EXP-006 trên Kaggle, cell 7 báo *WARM_START MISSING* vì user chưa Add input output EXP-003/EXP-005 **nhưng thật ra EXP-003 mặc định `KEEP_WEIGHTS=False` nên output của nó không hề chứa `best_checkpoint.pt`** — Add input cũng không tìm thấy. Giải pháp: `scripts/export_hybrid_warm_start.py` rút riêng `encoder.*` (M2 475MB/199 khóa, M4 0,4MB/32 khóa; verify warm start M4 missing=0) rồi push lên HF repo `Kichirou24/SemanticCodeCloneDetection` qua `--hf-repo`. Notebook cell 7 pull từ HF (`HF_TOKEN` nếu private), fallback quét `/kaggle/input` output cũ; thiếu checkpoint ⇒ **assert chặn train** để không tạo số không chính thức. |
| 27 | 1.26.0 | 2026-10-07 | DEC-029 §4 chốt tên HF repo **chính thức**: `Kichirou24/SemanticCodeCloneDetection` (public — SSH `git@hf.co` hoạt động trên máy chủ, không cần token). 2 file encoder đã push lên HF kèm README model card; notebook EXP-006 cell 7 và `export_hybrid_warm_start.py` đổi theo tên repo này. |
| 28 | 1.27.0 | 2026-10-07 | **Errata row 27:** repo HF đang **private** (anonymous read 401) — dòng "public, không cần token" sai. Local thao tác qua SSH `git@hf.co` (đã push thành công 2 commit). Trên Kaggle notebook vẫn lấy được vì cell 7 truyền `HF_TOKEN` (cell 1 đọc từ Kaggle Secrets) — đúng lời hứa "token của acc này dùng được". |
| 29 | 1.28.0 | 2026-10-17 | Thêm DEC-030: **dừng train, chuyển sang đánh giá + chọn phương pháp cuối (EXP-FINAL-001)**. Luật chọn = `validation.f1`, ngưỡng đóng băng theo run; winner vs á quân kiểm định cặp trên validation bằng McNemar + paired bootstrap tái dùng `src/evaluation/uncertainty.py`. Thêm `configs/final_pipeline.yaml`, `src/evaluation/leaderboard.py`, `scripts/final_pipeline.py` (`select`/`detect`), `tests/test_final_pipeline.py` (14 test), report `reports/014_final_selection.md`. Theo val F1 winner = M5-512 nhưng chỉ hơn M2-512 +0,0043 (McNemar p=0,081, bootstrap CI chạm 0 ⇒ `mixed`), ngưỡng 0,010 mỏng, thua test F1/AUC ⇒ **khuyến nghị vận hành M2-512**. |
| 30 | 1.29.0 | 2026-10-17 | Thêm DEC-031: **demo + báo cáo chuyển sang bản 512 cho M1/M2/M5** (`configs/demo_web.yaml` trỏ `results/exp00{2,3,6}_*_512`, `max_length: 512`, `best_checkpoint.pt`), thay mệnh đề "không đụng `demo_web.yaml`" của DEC-030 §4; M3/M4/B1 giữ nguyên vì không có biến thể 512. Chấm lại `data/custom_suite` (26 cặp): lỗi M2 3→2, M5 4→3 (sửa 2 dương giả ở `P2_nonclone_digitsum`), **không cặp nào đổi từ đúng sang sai**; thêm `score_custom_suite.py --merge` + `evidence_custom_suite.py --reuse-moss` để không ghi đè số MOSS service thật; guard `test_transformer_models_use_512_runs`. |
| 31 | 1.30.0 | 2026-10-17 | Thêm DEC-032: **bộ test mở rộng nhóm P5 — danh sách liên kết + hướng đối tượng** theo góp ý của thầy (test còn ngắn). 6 file C++ 59–71 dòng (`struct Node` + 4 hàm tự do ↔ `class IntList` + 4 phương thức cùng hành vi, 3 biến thể clone, 1 non-clone cùng khung) + 8 cặp mới (26 → 34 cặp): `P5_struct_class`, `P5_while`, `P5_scanf`, `P5_combo`, `P5_class_combo`, `P5_nonclone_palindrome`, `X_P1P5`, `X_P2P5`. Kết quả: **cả 5 model bắt đúng cặp struct↔class** (M1/M2/M5 ≈ 1,000; M3 0,915; M4 0,973) còn **MOSS bỏ sót cả hai** cặp clone dạng `class` (0,000); M3 đúng cả 8 cặp, M4 sai 3 (toàn dương giả); lỗi mới: `X_P1P5` bị M1/M2/M5 ≈ 1,000 vì hai bài cùng phép toán tổng. Cả 6 file vượt 512 subword token (520–683) nên M1/M2/M5 phải cắt tới 25 % code (M3/M4 không bị cắt). Công cụ: `score_custom_suite.py --pairs`, `evidence_custom_suite.py --models` + `fig_font`, `_latex_minipage(font=)`, và `.gitignore` mở ngoại lệ để `data/custom_suite` vào git. Báo cáo tuần mới `reports/[24 - 10] - Báo cáo.tex`. |

---

## DEC-028 — Baseline chính chuyển sang MOSS service thật (EXP-014); EXP-013 còn lại làm tham chiếu
- **Thay thế:** Quyết định 1 của DEC-026 (MOSS EXP-013 là baseline chính).
- **Bối cảnh (2026-10-07):** người dùng yêu cầu baseline phải là **MOSS thật**
  (`moss.stanford.edu`) qua `mosspy`, gồm cả demo. Đã gọi service thật cho toàn bộ
  val/test (user id qua env `MOSS_USER_ID`, không commit).
- **Quyết định 1 — baseline báo cáo là MOSS service thật (EXP-014),** runner
  `scripts/eval_moss_service.py`. Điểm cặp = `max(pct_a, pct_b)/100`; quy tắc vận
  hành `score > 0.01` ⇒ clone (không dò ngưỡng — tool dùng như công cụ đạo văn).
- **Quyết định 2 — EXP-013 (winnowing all-pairs local) còn lại làm tham chiếu.**
  Nó vẫn được dùng để (a) cross-check cài đặt và (b) fallback demo khi ứng dụng
  không có `MOSS_USER_ID` (deterministic, offline). Demo B1 dual-mode: service nếu
  có env, ngược lại local k=20,w=5.
- **Số liệu thật và bài học về top-N:** service MOSS chỉ trả **top-N match**
  (`show=1000`), không cho điểm mọi cặp — nên với bộ cặp cố định test, 11.000
  điểm gần như toàn 0 và chỉ bắt được 9/5.500 cặp dương (P 1,0 / R 0,0016 /
  F1 0,003). Đây không phải bug: `_safe_name` khớp 100% sid, parser đã test trên 2
  định dạng report, và bản local cùng thuật toán đạt F1 0,682 khi chấm được *mọi*
  cặp. Kết luận: MOSS giỏi ở vai trò "dò top-N clone gần trùng" nhưng đứng trong
  bài toán phân loại mọi cặp semantic clone Type-3/4 thì thiếu recall cơ bản —
  ghi thẳng vào báo cáo thay vì giấu.
- **Vận hành:** nộp MOSS tốn lượt (rate-limit) — chạy lại từ cache
  `artifacts/moss_cache` (val `http://moss.stanford.edu/results/6/3324282961870`,
  test `http://moss.stanford.edu/results/7/5556361909376`).
- **Impact:** bảng kết quả [06-10] ghi B1 = số EXP-014 kèm cảnh báo top-N; demo +
  registry + report đã cập nhật. M4 "hơn MOSS 0,097" không còn dùng nữa (đổi sang
  mốc EXP-013 tham chiếu).

[🔝 Quay lại mục lục](#top)

---

## DEC-029 — M5 hybrid soft fusion (EXP-006): fusion mức embedding, không OR/AND
- **Bối cảnh (2026-10-07):** user xem lại cách M1/M2 tiền xử lý (truncate 256,
  vứt phần sau) và hỏi M5 (M2+M4) nên xử lý sao. Số thật test: M1 CodeBERT
  P/R/F1 0,931/0,890/0,910 (AUC 0,973); M2 GraphCodeBERT 0,883/0,933/0,907
  (AUC 0,971); M3 ASTNN 0,760/0,894/0,821; M4 CPG-GNN 0,695/0,885/0,779.
  M1/M2 **truncate 60,5%** code nhưng vẫn thắng M3/M4 giữ 100% code ⇒ truncation
  không phải bottleneck thật trên POJ-104; không vội cải tiến mô hình text theo
  hướng segment/chunk trước khi kiểm bổ trợ của M4.
- **Quyết định 1 — M5 (EXP-006) là soft fusion mức embedding,** theo đúng
  `pair_architecture: concat_hA_hB_absdiff_mlp` của DEC-005 và head DEC-016:
  `h = [h_text (CLS 768) ; h_graph ([mean;max] 256)]` → `[hA ; hB ; |hA-hB|]` → MLP.
- **Quyết định 2 — KHÔNG dùng fusion nhị phân/rule:** audit bổ trợ trên test
  (script `scripts/audit_m2_m4_complementarity.py`, output
  `results/audit_m2_m4.json`): agreement 0,724, Cohen kappa 0,44; bảng
  both_ok/only_m2/only_m4/both_bad = 7.570/2.377/662/391. M4 sửa 662/1.053 lỗi
  M2 (62,9%) nhưng làm hỏng 2.377/9.947 cặp đúng M2 (23,9%), net −1.715; quy tắc
  OR F1 0,8087 và AND F1 0,8795 đều kém M2 (0,9069) ⇒ **loại OR/AND và mọi dạng
  score-level rule cứng**.
- **Quyết định 3 — baseline của M5 chính là logistic mềm trên (s_M2, s_M4):**
  fit trên validation → test F1 0,9217 > M2 0,9069 (+1,5đ; P/R 0,905/0,939), val
  F1 0,9463 (thr 0,535). Nhưng ROC-AUC 0,9683 < M2 0,9713 ⇒ thắng chỉ ở vùng
  ngưỡng vận hành (operating region) chứ không thắng mặt bằng xếp hạng. **Báo cáo
  bắt buộc nêu đúng điều này**, không được viết “hybrid thắng” chung chung.
- **Quyết định 4 — warm start encoder rồi joint fine-tune:** 1) copy trọng số
  `encoder.*` từ M2 (EXP-003) và M4 (EXP-005) — nạp cùng biểu diễn mà audit đã
  đo; 2) joint fine-tune LR **2e-5**, 5 epoch, patience 2 (LR to sẽ phá biểu diễn
  đã hội tụ). Giữ tùy chọn `freeze_encoders` như bản rẻ (chỉ train head MLP).
  **Kênh phân phối checkpoint = Hugging Face Hub** (bổ sung 2026-10-07, xem row
  26): checkpoint `*.pt` gitignored + output EXP-003 trên Kaggle không giữ
  checkpoint (`KEEP_WEIGHTS=False`) ⇒ sinh bằng `scripts/export_hybrid_warm_start.py`
  (chỉ `encoder.*`: M2 475MB, M4 0,4MB) rồi pull về từ HF repo
  `Kichirou24/SemanticCodeCloneDetection` ở cell 7 của notebook.
- **Quyết định 5 — giữ `max_length 256` cho M5** (đóng băng từ DEC-016/022) để so
  sánh sạch với M2. Segment-GRU / nâng 510 chỉ làm sau khi có kết quả M5.
- **Loại bỏ:** fusion OR/AND/metric-level; blending score đơn thuần (Cardinal) làm
  mô hình chính; thay tokenizer/đổi recipe giữa chừng.
- **Hệ quả:** tầng tiền xử lý M1/M2 giữ nguyên; M5 tái dùng cache CPG của EXP-005
  (không parse Joern lại); code: `src/models/hybrid/model.py`,
  `src/experiments/exp006_hybrid.py`, config `configs/exp006_hybrid.yaml`, notebook
  `notebooks/exp006_hybrid_kaggle.ipynb`.
- **Impact:** EXP-006 = READY_FOR_KAGGLE_TRAIN; thêm 5 test model (`tests/
  test_exp006_hybrid.py`) không cần GPU/HF. Nếu M5 vẫn không vượt logistic ⇒ M4
  thừa; nếu vượt ⇒ cải tiến tiếp theo là nâng độ dài text, không phải phân loại
  độ phức tạp.

[🔝 Quay lại mục lục](#top)

---

## DEC-030 — Dừng train; chọn phương pháp cuối bằng val F1 + pipeline EXP-FINAL-001
- **Bối cảnh (2026-10-17):** đã có 9 run hoàn thành (B0/B1, M1/M2/M5 ở cả 256 và
  512, M3, M4). Các bản 512 cho trần thực dụng F1 ≥ 0,94; phần chênh còn lại nằm
  trong nhiễu. Người dùng quyết định **dừng huấn luyện**, chuyển sang đánh giá để
  chọn phương pháp tốt nhất rồi dựng một pipeline cuối hoàn chỉnh.
- **Quyết định 1 — luật chọn là `validation.f1`,** không theo test. Test chỉ để
  báo cáo (giữ nguyên DEC-022). Chỉ run `role: candidate` được chọn; baseline
  (MOSS) nằm trong bảng để đối chiếu nhưng không thể "thắng".
- **Quyết định 2 — ngưỡng đóng băng theo từng run** từ `threshold.json`; pipeline
  tuyệt đối không dò lại ngưỡng trên test (dò lại = rò rỉ nhãn, CI hẹp giả tạo).
- **Quyết định 3 — lượng hoá độ bất định bằng kiểm định cặp, không so hai cột
  ±SD:** winner vs á quân so trên **chính tập validation** bằng McNemar (exact
  binomial) + paired bootstrap CI 95% của ΔF1, **tái dùng `src/evaluation/
  uncertainty.py`** (không cài đặt lại; module đã có test chống tràn và dùng đúng
  ngưỡng riêng mỗi model). Hai kiểm định bất đồng ⇒ gắn nhãn `mixed`, không chọn
  bừa.
- **Quyết định 4 — dựng pipeline thành 1 CLI 2 chế độ** (`scripts/final_pipeline.py`):
  `select` (xếp hạng + chọn, chạy được không cần torch/GPU/Joern) và `detect`
  (chấm một cặp file bằng model đã chọn, ngưỡng đọc từ `threshold.json`). Khai báo
  ở `configs/final_pipeline.yaml`; **không đụng `configs/demo_web.yaml`** (demo vẫn
  6 model như cũ).
- **Kết quả:** theo val F1, **M5-512 thắng** (0,9379 vs M2-512 0,9337) nhưng pipeline
  nêu 4 cảnh báo: ngưỡng 0,010 rất mỏng; chênh +0,0043 (< 0,01); **McNemar p = 0,0814
  (chưa ý nghĩa) trong khi paired bootstrap CI95 [+0,0004, +0,0083] vừa chạm khác 0
  ⇒ bất đồng**; và M2-512 mạnh hơn ở test F1 (0,9550) lẫn AUC (0,9912).
- **Khuyến nghị vận hành (đã chốt):** dùng **M2-512** cho pipeline cuối (AUC cao
  nhất, ngưỡng khoẻ 0,66, không cần JVM lúc suy luận) — đặt thẳng
  `configs/final_pipeline.yaml → inference.default_model: M2_512`. `select` vẫn báo
  M5-512 là winner theo protocol kèm đủ cảnh báo; gọi M5-512 khi cần bằng
  `--model M5_512`.
- **Hệ quả:** thêm `src/evaluation/leaderboard.py`, `scripts/final_pipeline.py`,
  `tests/test_final_pipeline.py` (14 test, không cần GPU), report
  `reports/014_final_selection.md`, artifact `results/final_pipeline/`. Biến `k` của
  kế hoạch train (Hybrid V3, CPG nới node) **tạm gác** — chỉ mở lại nếu cần bằng
  chứng mạnh hơn cho lựa chọn.

[🔝 Quay lại mục lục](#top)

---

## DEC-031 — Demo và báo cáo dùng bản 512 cho M1/M2/M5 (thay bản 256)
- **Bối cảnh (2026-10-17):** khi rà lại hình kiến trúc pipeline M2, người dùng phát hiện
  phần minh hoạ/báo cáo vẫn chấm bằng checkpoint **256**, trong khi bản **512** đã có từ
  12-10/15-10 và tốt hơn ở cả ba chỉ số test (M1 0,910→0,941; M2 0,907→0,955; M5
  0,913→0,952). Giữ bản cũ nghĩa là báo cáo đang mô tả một model không còn là bản tốt
  nhất của nhóm.
- **Quyết định 1 — `configs/demo_web.yaml` cho M1/M2/M5 trỏ vào run 512**, `max_length: 512`,
  checkpoint `best_checkpoint.pt`: M1 → `results/exp002_codebert_poj104_512/`,
  M2 → `results/exp003_graphcodebert_poj104_512/`, M5 → `results/exp006_hybrid_poj104_512/`.
  **Thay thế** mệnh đề "không đụng `configs/demo_web.yaml`" của DEC-030 §4 — mệnh đề đó
  chỉ để giữ demo im lặng trong lúc thêm pipeline cuối, không phải để đóng băng demo ở
  bản 256.
- **Quyết định 2 — M3/M4/B1 giữ nguyên.** Ba phương pháp này không có biến thể 512: M3 làm
  việc trên cây AST (theo câu lệnh), M4 trên node CPG, B1 là so khớp winnowing — đều không
  phụ thuộc `max_length` của encoder text. Đổi chúng chỉ để "cho đủ" là tạo biến giả.
- **Quyết định 3 — ngưỡng vẫn không dò lại:** mỗi model đọc ngưỡng từ `threshold.json`
  của chính run (M1-512 0,210; M2-512 0,660; M5-512 0,010), đúng DEC-022 / DEC-030 §2.
- **Cách chấm lại:** `scripts/score_custom_suite.py --models M1 M2 M5 --merge` (giữ nguyên
  B1/M3/M4). Thêm hai cờ để việc chấm lại không phá artifact cũ:
  `score_custom_suite.py --merge` và `evidence_custom_suite.py --reuse-moss` — máy chạy
  không khai `MOSS_USER_ID` nên B1 sẽ rơi vào fallback winnowing local, ghi đè là mất số
  MOSS service thật của báo cáo.
- **Kết quả trên `data/custom_suite` (26 cặp), so với bản 256:**
  - `P1_combo` (clone): M1 0,999999 · M2 **1,000** · M5 0,999998 — cả 5 vẫn đúng.
  - `P3_nonclone_sum` (non-clone): M2 0,110 → **0,005**, M5 0,005 → **0,000** — vẫn đúng.
  - `P1_nonclone_fact` (non-clone): M1 1,000 → **0,999**, M2 1,000 → **0,991**,
    M5 1,000 → **0,723** — cả 5 vẫn sai (đều trên ngưỡng).
  - `P2_nonclone_digitsum` (non-clone): M2 0,505 → **0,001**, M5 0,981 → **0,000**
    — **sửa được 2 lỗi dương giả**.
  - `X_P1P4`: M1 0,577 → 0,000, M2 0,385 → 0,000 (trước đó đã đúng, nay dứt khoát hơn).
  - `X_P2P4`: M5 0,542 → 0,027 — **vẫn sai** (vẫn trên ngưỡng 0,010); đây là lỗi dương
    giả duy nhất còn lại ngoài 2 cặp bẫy skeleton.
  - Tổng số cặp sai trên 26 cặp: M2 3→**2**, M5 4→**3**; M1 (2), B1 (16), M3 (10), M4 (10)
    không đổi. Không có cặp nào đổi từ đúng sang sai.
- **Hệ quả:** hình minh hoạ §4 của `reports/[17 - 10] - Báo cáo.tex` sinh lại từ
  `workspace/demo_suite_*.tex`; `tests/test_demo_registry.py` thêm
  `test_transformer_models_use_512_runs` để không ai lùi demo về 256; bản 256 vẫn nằm
  trong `results/` và bảng huấn luyện/test của báo cáo vẫn in cả hai để đối chiếu.

[🔝 Quay lại mục lục](#top)
