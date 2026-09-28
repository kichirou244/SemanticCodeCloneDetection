# 011 — EXP-002 M1 CodeBERT trên POJ-104 (COMPLETED)

Ngày chốt kết quả: 2026-09-28. Báo cáo này thay thế `reports/010_codebert_audit.md`
(audit code, **không** phải kết quả) và là nguồn số liệu chính thức của M1 cho RQ1.
Phiên bản `v1.14.0` (2026-09-28) — thêm 11.13 về quy trình tải/đóng gói checkpoint
bản nhẹ `best_weights_only.pt` (DEC-022).

## 11.1 Experiment Identity

- Experiment ID: **EXP-002** · Model: **M1 CodeBERT** (`microsoft/codebert-base`) · Dataset: POJ-104 (C/C++).
- Manifest: `data/manifests/poj104.json` — sha256 `909615d8…4ebaf84e`, `leakage_passed: true`.
- Config: `configs/exp002_codebert.yaml` — sha256 `a482ccbb…12e78ff544` (bản khớp đúng run; xem 11.14).
- Code: git `2885e02` (monolithic `exp002_codebert.py`; sau khi train đã refactor sang
  `transformer_pair_runner` theo DEC-020 — hành vi giữ nguyên, xem 11.13).
- **Train**: Kaggle, Tesla T4, `torch 2.10.0+cu128`, `transformers 5.0.0`, Python 3.12.13,
  `wall_clock 2026-09-27T14:38:25+00:00`.
- **Eval**: local, RTX 5060 Laptop 8.5GB, fp32, `--mode evaluate` (5 phút 16 giây) — xem 11.11.
- Artifact: `results/exp002_codebert_poj104/` (`best_checkpoint.pt` 1,44GB không commit — `*.pt` bị gitignore).

## 11.2 Objective

RQ1 hỏi representation mạnh có vượt được lexical baseline không. EXP-001 (TF-IDF + cosine)
đạt test F1 **0.673** với đặc trưng lỗi là FP-dominant (3.834 FP vs 770 FN). EXP-002 trả lời
riêng một câu: **M1 CodeBERT cải thiện bao nhiêu so với B1 trên cùng split, cùng pairs,
cùng protocol chọn threshold**? Kết luận về M1 vs M2 (EXP-003) **không** rút ra ở đây.

## 11.3 Dataset

- 51.991 canonical samples (dedup 1.009); split theo **problem** 83/10/11.
- Pairs 1:1 — train **332.000** (166k/166k), validation 10.000, test 11.000.
- Trùng problem giữa các split: 0 (kiểm tra ở `reports/008_final_leakage_audit.md`).
- Pair **dương** = 2 sample cùng problem; **âm** = 2 problem khác nhau. Nhãn sinh từ
  partition sẵn có, không dùng nhãn của bất kỳ mô hình nào.

## 11.4 Model & Architecture

- Encoder: `microsoft/codebert-base` (RoBERTa-base 125M, vocab 50265) — **không** fallback
  random nếu download lỗi (fail-fast).
- Siamese **shared**: một encoder xử lý cả hai phía, lấy `[CLS]` (`last_hidden_state[:,0,:]`, 768-d).
- Pair head (DEC-016): `[hA ; hB ; |hA−hB|]` = 2304-d → `Linear(768) → Tanh → Dropout(0.1) → Linear(1)`.
- Tổng tham số **126.416.641** (encoder + head đều trainable), loss `BCEWithLogitsLoss` trên raw logit.
- Tokenizer: `AutoTokenizer` của checkpoint (không regex), `max_length = 256`, dynamic padding.

## 11.5 Training Protocol

| Tham số | Giá trị |
| :--- | :--- |
| epochs / seed | 3 / 42 |
| batch_size × grad_accum | 16 × 2 → 32 pair/update |
| learning_rate / weight_decay | 2e-5 / 0.01 |
| warmup_ratio (linear, **per optimizer update** — DEC-017) | 0.1 |
| max_grad_norm / mixed_precision | 1.0 / true (AMP) |
| Tổng optimizer update | 31.125 |
| Thời gian train | 25.306 s = **7,03 h** |
| Kết thúc vì | `epochs_exhausted` (chưa chạm budget 9 h) |
| **Best epoch** | **1** |

Lịch sử từng epoch (`training_history.csv`):

| Epoch | train_loss | val_loss | val_f1 | val_roc_auc | lr | epoch_time_s |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1635 | 0.3769 | 0.91909 | 0.97601 | 1.481e-05 | 8.534 |
| 1 | 0.0220 | 0.3937 | **0.91912** | 0.97701 | 7.407e-06 | 8.399 |
| 2 | 0.0076 | 0.4416 | 0.91156 | 0.97741 | 0.0 | 8.370 |

Bằng chứng F-02 của DEC-017 đã được sửa trong **run thật**: `run_metadata.json` ghi
`device_requested: cuda`, **`device_actual: cuda:0`**, `model_device: cuda:0`,
`cuda_device_name: Tesla T4`, `cuda_available: true` — metadata cũ hardcode
`device: cpu` không còn là bằng chứng đáng tin nữa (dòng `device: cpu` vẫn còn do
`exp001_tfidf.run_metadata()` hardcode; xem 11.14 threats).

## 11.6 Threshold Protocol

- Grid 0.01 trên **validation**, tối ưu F1 → **0.59** (val F1 tại ngưỡng này = 0.9200).
- Freeze ngưỡng rồi mới chạy trên test **đúng một lần**; test label không tham gia chọn ngưỡng.
- Threshold lưu ở `results/exp002_codebert_poj104/threshold.json`.

## 11.7 Results

| Split | Prec. | Recall | F1 | ROC-AUC | PR-AUC | Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.9231 | 0.9170 | 0.9200 | 0.9770 | 0.9791 | 0.9203 |
| **Test** | **0.9309** | **0.8898** | **0.9099** | **0.9729** | **0.9742** | 0.9119 |

Test (n = 11.000): TP 4.894 · TN 5.137 · **FP 363** · **FN 606**.

Validation ≈ test (F1 chênh 0.010) → ngưỡng generalize tốt, **không** overfit threshold
(đúng kỳ vọng protocol §14).

## 11.8 So sánh với EXP-001 (cùng split, cùng pairs)

| Mô hình | P | R | F1 | ROC-AUC | PR-AUC | FP | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP-001 B1 TF-IDF | 0.552 | 0.860 | 0.673 | 0.690 | 0.704 | 3.834 | 770 |
| **EXP-002 M1 CodeBERT** | **0.931** | **0.890** | **0.910** | **0.973** | **0.974** | **363** | **606** |
| Δ | +0.379 | +0.030 | **+0.237** | +0.283 | +0.270 | −3.471 | −164 |

- Precision tăng 0.379: model học được khác biệt **ngữ nghĩa**, không còn coi boilerplate C
  (`#include`, `int main`, vòng lặp IO) là bằng chứng clone — đúng lỗi FP đặc trưng của B1.
- Recall gần như giữ nguyên (0.860 → 0.890) **và** FP giảm 10,6 lần ⇒ đây không phải
  "hy sinh recall để đổi precision".
- Hướng lỗi **đảo ngược**: B1 FP-dominant (FP/FN = 5,0) → M1 FN-dominant (FP/FN = 0,60).
  Ở ngưỡng tối ưu F1, M1 còn 606 FN — đây là phần dễ sửa nhất bằng cách hạ ngưỡng
  (đánh đổi precision), xem 11.9 vì sao FN lại tập trung ở code dài.

## 11.9 Error Analysis (đo trên `predictions/test_scores.jsonl`, ngưỡng 0.59)

Kiểm tra ngữ nghĩa nhãn trên top-100 lỗi mỗi loại: **FN 100/100 cùng problem**,
**FP 100/100 khác problem** — nhãn cặp đúng như thiết kế, lỗi là lỗi mô hình chứ không
phải lỗi dữ liệu.

**FN tập trung ở code dài** — median độ dài cạnh dài nhất của cặp FN là **1.109 ký tự**
so với median toàn dataset **528**:

| Quartile độ dài (ký tự) | n | FP | FN | Tỉ lệ lỗi |
| :--- | ---: | ---: | ---: | ---: |
| Q1 (160–541) | 2.750 | 48 | 95 | **5,2 %** |
| Q2 (541–753) | 2.750 | 86 | 134 | 8,0 % |
| Q3 (753–1.007) | 2.750 | 108 | 156 | 9,6 % |
| Q4 (1.007–5.042) | 2.750 | 121 | 221 | **12,4 %** |

Tỉ lệ lỗi tăng **2,4×** từ quartile ngắn nhất sang dài nhất, và **FN tăng 2,3×** (95 → 221).
Đây là hệ quả trực tiếp của truncation ở `max_length = 256` (11.10): cặp FN dài bị cắt
mất phần chứa biến thể khác nhau, hai phía còn lại gần như giống nhau ⇒ score thấp.

**FP thì ngược lại**: median cạnh dài nhất 861 ký tự (thấp hơn FN 1.109) — cặp âm bị chấm
cao là cặp *ngắn*, dùng chung skeleton. Chi tiết 100 FP + 100 FN:
`results/exp002_codebert_poj104/error_cases.jsonl` (lưu ID + score; truy code qua
`data/processed/poj104/samples.jsonl`).

## 11.10 Overfitting & Early Stopping

- **Có overfit**, nhưng không nằm ở chỗ ta đo: `train_loss` giảm 21× (0.1635 → 0.0076)
  trong khi `val_loss` **tăng đều** 17 % (0.3769 → 0.4416).
- Sức phân biệt gần như không đổi: `val_roc_auc` 0.9760 → 0.9770 → 0.9774 (vẫn tăng),
  `val_f1` 0.91909 → 0.91912 → 0.91156. ⇒ phần tăng của `val_loss` là **độ tự tin thừa
  (calibration)**, không phải mất khả năng phân biệt. Checkpoint được chọn theo
  `val_f1` nên epoch 2 (overfit rõ nhất) **không** được dùng để evaluate.
- **Early stopping không bao giờ kích hoạt** và đó là lỗi thiết kế, không phải may mắn:
  `patience = 2` nhưng không có `min_delta`, nên best_f1 epoch 1 hơn epoch 0 đúng
  0.00003 vẫn bị tính là "improved". Hậu quả đo được: **epoch 3 lãng phí ~2,3 h** (~33 % thời gian train).
- **Không sửa config cho run này** — xem DEC-022: `max_length` và `min_delta` phải giữ
  giống nhau cho M1/M2, sửa riêng EXP-002 lúc này sẽ phá vỡ tính công bằng của RQ1.

## 11.11 Truncation — threat lớn nhất của kết quả

| Split | n code | n bị truncate | Tỉ lệ |
| :--- | ---: | ---: | ---: |
| train | 41.491 | 32.618 | **78,61 %** |
| validation | 5.000 | 3.676 | **73,52 %** |

Đo bằng tokenizer thật, per-code, trước khi cắt (định nghĩa khớp F-08 của audit 010).
Median code ~150 token nên phần lớn dataset **vẫn vừa** 256 — nhưng 1/4 quartile dài là
vùng mà mô hình mù hoàn toàn, và 11.9 cho thấy lỗi tập trung đúng ở đó. Đây là lý do F1
0.910 **không** nên đọc là "giải quyết được clone detection ngoài đời".

## 11.12 Runtime

| Giai đoạn | Thiết bị | Thời gian |
| :--- | :--- | ---: |
| Train (3 epoch, 31.125 update) | Kaggle Tesla T4 | 25.306 s (7,03 h) |
| Train tối đa | — | 3 × ~8.400 s ≈ 7,0 h/epoch |
| Eval (val 10k + test 11k pair, fp32) | local RTX 5060 8,5GB | **316 s (5 m 16 s)** |

Ghi chú: `eval_time_s` trong `EVAL_COMPLETE.json` của lần chạy này báo `0.096` —
nó chỉ đo thời gian **ghi artifact**, không tính inference (timer khởi tạo sai chỗ). Đã
sửa ở `transformer_pair_runner.evaluate_model` và `exp004_astnn.evaluate_model`; số
316 s ở trên là đo ngoài (`Stopwatch`), artifact của run này vẫn giữ giá trị cũ. Sau khi
sửa, lần evaluate chạy lại từ checkpoint bản nhẹ (11.13) báo `eval_time_s = 298,5 s`,
khớp với đo ngoài 316 s (chênh do warm-up cuối epoch).

## 11.13 Reproducibility

Train (chỉ chạy trên Kaggle, ~7 h):

```bash
python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml \
  --mode train --device cuda --time-budget-s 32400
```

Evaluate (**chạy ngay trên Kaggle, 5 phút — cell `--mode evaluate` của notebook**):

```bash
python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml \
  --mode evaluate --device cuda
```

**Không cần tải checkpoint về máy.** Sau khi evaluate xong trên Kaggle, tải về **chỉ
metrics + predictions** (vài MB) là đủ: `predictions/{validation,test}_scores.jsonl` cho
phép tính lại **mọi** metric, threshold và error analysis mà không cần model. Quyết định
này (DEC-023) đảo hướng của DEC-022 §2 — cùng nguyên tắc, hướng ưu tiên khác: 6 phút
GPU trên Kaggle rẻ hơn tải 0,5–1,44GB rồi chạy lại local.

Nếu vẫn muốn chạy evaluate ở máy khác (ví dụ máy không có GPU rảnh), thì chỉ cần
`best_weights_only.pt`, **không** phải checkpoint đầy đủ:

| File | Nội dung | Kích thước | Cần tải về? |
| :--- | :--- | ---: | :--- |
| `best_checkpoint.pt` | model + optimizer + scheduler + RNG state | 1,44GB | ✗ (chỉ để resume trên Kaggle) |
| `last_checkpoint.pt` | như trên, của epoch cuối | 1,44GB | ✗ (chỉ để resume) |
| `best_weights_only.pt` | chỉ `model_state` + `epoch` + `best_f1` | **0,51GB** | △ chỉ khi muốn evaluate ở máy khác |

- `best_weights_only.pt` chỉ chứa tensor + số nên `load_eval_weights()` nạp nó bằng
  `torch.load(..., weights_only=True)` (không unpickle object tuỳ ý); checkpoint đầy đủ
  vẫn phải `weights_only=False` vì có `np.random.get_state()`. Cấu trúc giữ nguyên
  (`model_state`) nên `evaluate_model` dùng được mà không sửa code. Runner ưu tiên
  checkpoint đầy đủ nếu còn, không thì tự fallback sang bản nhẹ.
- **Đã kiểm chứng trên chính artifact của lần chạy này**: xoá `best_checkpoint.pt`,
  chạy `--mode evaluate` lại ⇒ metric **giống hệt** (F1 `0.9099191224319048`, TP 4894,
  TN 5137, FP 363, FN 606), `eval_time_s = 298,5 s` (đúng thời gian thật sau fix timer
  ở 11.12). Quy trình cũ (`--exclude='*.pt'`) khiến archive **không còn gì để evaluate**,
  buộc tải 1,44GB; đã đổi sang `--exclude='*checkpoint.pt'` để giữ bản nhẹ.
- Checkpoint (`*.pt`) **không** commit — `.gitignore` loại `*.pt`. Khôi phục train bằng
  `--mode resume` (cần `last_checkpoint.pt` từ version Kaggle đã save) hoặc train lại.
- `config.yaml` + `manifest.json` copy trong thư mục result là snapshot của đúng run; hash
  cũng nằm trong `run_metadata.json` (`config_sha256`, `manifest_sha256`).
- `predictions/{validation,test}_scores.jsonl` (11k + 10k dòng) đủ để tính lại **mọi**
  metric, threshold và error analysis mà không cần model. Mọi số trong 11.7–11.9 tính
  lại được từ hai file này.
- Eval chạy trên GPU khác máy train (RTX 5060 vs T4, đều fp32) — sai số chỉ ở mức
  floating-point, nhưng **predictions/ mới là nguồn số chuẩn**, không phải con số in ra.

## 11.14 Threats / Limitations

- **Truncation 78,6 %** ở 256 token → kết quả không đại diện cho code dài; lỗi tăng 2,4×
  ở quartile dài nhất (11.9).
- **Mẫu pair dương không độc lập**: 166.000 pair dương sinh từ chỉ 41.491 problem train, nên
  nhiều cặp gần trùng ⇒ F1 có thể lạc quan so với deployment thực (dữ liệu mới, problem mới).
- **`device: cpu` vẫn nằm trong `run_metadata.json`** (hardcode ở `exp001_tfidf.run_metadata()`,
  F-02 audit 010 chỉ sửa nửa vì: các trường `device_actual`/`cuda_device_name` mới là
  bằng chứng đúng). Đã quyết định không sửa giữa chừng vì không ảnh hưởng số liệu; EXP-003
  nên sửa dứt điểm.
- **Early stopping vô hiệu** (không `min_delta`) → tốn ~2,3 h/epoch cuối không đổi kết quả.
- **Chỉ classification**; chưa có retrieval metric (đúng §13, không ép).
- **Một run, một seed (42)** — chưa có variance estimate, chưa kết luận được khác biệt có ý nghĩa thống kê.
- **So sánh M1 vs M2 chưa hợp lệ** cho tới khi EXP-003 chạy cùng recipe (DEC-020); kết luận
  ở đây chỉ là M1 **vượt** B1.

## 11.15 Conclusion

M1 CodeBERT đạt **test P 0.9309 / R 0.8898 / F1 0.9099 (ROC-AUC 0.9729)**, vượt baseline
lexical EXP-001 **+0.237 F1** nhờ giảm FP 3.834 → 363 mà recall không giảm. Bằng chứng
rằng representation ngữ nghĩa vượt lexical trên cùng split/pairs/protocol — trả lời
"phần" của RQ1; phần còn lại (M2 có tốt hơn M1 không) chờ EXP-003.

Hai kết luận phụ đáng ghi: (1) lỗi M1 **tập trung ở code dài** (12,4 % vs 5,2 % theo
quartile độ dài) — nguyên nhân có thể đo được, không phải suy đoán; (2) overfit biểu hiện
ở calibration chứ không ở ranking, và early stopping hiện tại không bắt được nó.
Cả hai đều là hạng mục cho phase ablation, **không** sửa giữa chừng để giữ M1/M2 cùng recipe.
