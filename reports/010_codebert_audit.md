# 010 — EXP-002 CodeBERT audit (AUDIT & FIX, không phải paper result)

Ngày: 2026-09-21. Run audit: `results/exp002_codebert_poj104_invalid_run/`
(git 1a94cdc, config sha256 cb6c2ef5...). Run này INVALID — không dùng làm M1 result.

## 38.1 Current run

- Lệnh: `python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml --mode train --device cuda`
- Config: codebert-base, [hA;hB;|hA-hB|]+MLP(Tanh), max_length 256, epochs 3,
  batch 16, accum 2, lr 2e-5, warmup 0.1, AMP true.
- Metadata: `device_requested=cuda`, `device=cpu`, torch 2.10.0+cu128, train 27964 s.
- Metrics: val P/R/F1 0.50005/1.0/0.66671, ROC-AUC 0.49673; test ROC-AUC 0.52059,
  TN=0/1, FP=5500/4999. Training history: loss 0.7035→0.6970→0.6943;
  val_f1 0→0.0056→0.4242; **lr ghi nhận 6.4e-09 / 1.3e-08 / 1.9e-08**.

## 38.2 Findings

| ID | Severity | Evidence | Root cause | Impact | Fix | Verification |
|----|----------|----------|------------|--------|-----|--------------|
| F-01 | CRITICAL | `training_history.csv`: lr ~1e-8 thay vì 2e-5 | `scheduler.step()` gọi 1 lần/epoch trong khi `total_steps = steps_per_epoch*epochs` (=31125) tính theo optimizer-update; scheduler warmup-linear nên lr ≈ 2e-5×step/3112 | Model hầu như không học → loss ~random, classifier collapse (predict ~all-positive) | DEC-017: scheduler.step() mỗi optimizer update trong `run_batches` (qua `_optimizer_step`); `steps_per_epoch` dùng ceil | Diagnostic CPU: loss 1.014→0.858 sau 2 epoch, scores phân tán (std 0.12) thay vì collapse |
| F-02 | HIGH | `run_metadata.json`: `device="cpu"` dù `device_requested="cuda"` | `run_metadata()` trong `exp001_tfidf.py` hardcode `"device": "cpu"`; EXP-002 tái dùng mà không ghi đè | Không thể xác minh GPU từ artifact; metadata gây hiểu lầm | `collect_device_metadata()`: ghi `device_requested/device_actual/model_device/cuda_available/cuda_device_name/gpu mem`; `assert_device_setup()` + assert input/label/logit device → FAIL FAST | Code path mới; GPU thực tế chờ user chạy diagnostic |
| F-03 | LOW | `steps_per_epoch` dùng `//` (floor) | Bỏ batch dư cuối khi tính total_steps | total_steps lệch nhẹ vs thực tế | Dùng `math.ceil` | — |
| F-04 | INFO | val_f1 epoch 2 = 0.42 nhưng eval cuối F1(val)=0.667 ở thr 0.47 | Không phải bug: history đo ở thr 0.5, eval cuối tune thr trên val (đúng protocol) | Không ảnh hưởng tính đúng đắn | Giữ nguyên threshold-trên-val | — |

Không phải bug (đã xác minh): F-05 pair-protocol — `test-neg` tham chiếu
`sample_id` prefix `poj104-train-…/validation-…` nhưng trường `split` thực tế
đều = `test`, problem 18/79/83 đều thuộc test problem set
(`data/splits/poj104/test_problems.json`); prefix chỉ là tên shard download,
không phải split. Không leakage. F-06 checkpoint/threshold/eval: best checkpoint
theo val-F1 được load cho eval, threshold chỉ trên val — đúng protocol.
F-07 pooling/pair/loss: CLS `hidden[:,0,:]`, concat 2304-dim, BCEWithLogitsLoss
trên raw logits, encoder+head đều trainable (126,416,641 params) — đúng DEC-016.
F-08 truncation 78.6%/73.5% đo sau tokenizer thật (per-code, trước truncate ở
`max_length=256`) — định nghĩa rõ, chấp nhận được, median code ~150 tokens.

## 38.3 GPU audit

- CLI `--device cuda` chỉ tạo `torch.device("cuda")`; không có assert nào trước fix.
- Metadata cũ không chứng minh được CPU hay GPU (hardcode). Thời gian 7.77h
  không kết luận được.
- Sau fix: `assert_device_setup` (cuda available + model trên cuda) ở train,
  evaluate, diagnostic; assert input/label/logit device mỗi batch; metadata đầy đủ.
- Môi trường agent hiện tại: `torch.cuda.is_available()=True`,
  GPU `NVIDIA GeForce RTX 5060 Laptop GPU` (thấy trong diagnostic CPU run).

## 38.4 Training audit

- Forward: siamese shared CodeBERT → CLS → [hA;hB;|hA-hB|] (2304) → MLP → 1 logit. Đúng DEC-016.
- Loss: BCEWithLogitsLoss, labels float32, không sigmoid kép.
- Backward/step/zero_grad + AMP GradScaler + clip_grad_norm: đúng.
- Optimizer nhận toàn bộ `model.parameters()` (encoder+head), lr 2e-5, wd 0.01.
- Scheduler: **sai trước fix (F-01), đã sửa**.
- Checkpoint: save mỗi epoch + best theo val-F1; eval load `best_checkpoint.pt`. Đúng.

## 38.5 Data audit

- Labels batch kiểm tra trong diagnostic: unique {0,1}, balanced 32/32.
- Alignment code_a/code_b/label: `collate_pair_batch` build từ cùng `chunk` order;
  eval scores zip với cùng `refs` order — không shuffle lệch.
- Split semantics: pairs sinh độc lập mỗi split từ samples của split đó
  (`PairGenerator.generate(samples, split)`); cross-split sample reuse không tồn tại
  (F-05 giải thích prefix shard).
- Leakage: problem split 83/10/11 disjoint; không sample nào thuộc 2 split.

## 38.6 Tokenization audit

- Single source of truth: `cfg[tokenizer][max_length]=256`, truyền thẳng vào
  `tokenize_side(..., max_length)` cho cả 2 phía (per-code, không per-pair).
- Warning "702 > 512" đến từ `tokenizer.model_max_length=512` của checkpoint khi
  đo `truncation=False` — hành vi đo cố ý, không phải lỗi config.
- Truncation rate = fraction codes có tokenized length > 256 (đo sau tokenizer thật).

## 38.7 Diagnostic result — PASS (CPU logic)

`--mode diagnostic --device cpu`: 64 pairs balanced, 2 epochs.
`DIAGNOSTIC: PASS` — cuda_available True, GPU name OK, trainable 126,416,641
(encoder 124,645,632 + head 1,771,009), gradients finite, params updated,
logits vary (std 0.124), loss finite và giảm (1.014→0.858).
GPU-device path (model/input trên cuda:0, allocated>0) chờ user chạy diagnostic GPU.

## 38.8 Decision

**READY_FOR_FULL_RETRAIN** (sau khi user chạy diagnostic GPU PASS).
Run cũ archived tại `results/exp002_codebert_poj104_invalid_run/` + `INVALID_RUN.json`.
