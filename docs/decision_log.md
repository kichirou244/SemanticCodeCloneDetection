<a id="top"></a>

# Decision log (decision_log.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.13.0 |
| **Ngày cập nhật** | 2026-09-28 |
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
