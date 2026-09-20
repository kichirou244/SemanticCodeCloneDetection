# 009 — EXP-001 Baseline TF-IDF + Cosine trên POJ-104 (COMPLETED)

## 9.1 Experiment Identity
- Experiment ID: EXP-001 · Model: B1 TF-IDF + Cosine · Dataset: POJ-104 (C/C++)
- Manifest: `data/manifests/poj104.json` (phase1-v1, leakage passed, problem split)
- Config: `configs/experiments/exp001.yaml` (seed 42, max_features 20000,
  batch 4096, threshold validation_f1 grid 0.01, top_k 100)
- Date: 2026-09-20 · Env: Windows 11, Python 3.12.10, CPU-only (xem
  `results/exp001_tfidf_poj104/run_metadata.json` cho git/config/manifest hash)

## 9.2 Objective
Baseline lexical cổ điển, độc lập hoàn toàn M1–M5: đo xem tương đồng từ vựng
đơn thuần phân biệt được clone ngữ nghĩa tới đâu, làm mốc cho RQ1
(TF-IDF vs Transformer vs AST vs CPG).

## 9.3 Dataset
- 51.991 canonical samples (dedup 1.009); split 83/10/11 problems.
- Pairs 1:1: train 332.000 (166k/166k), val 10.000, test 11.000.
- Fit TF-IDF chỉ trên 41.491 train samples; val/test chỉ transform (§6).

## 9.4 Preprocessing
- Raw code giữ nguyên; không rename identifier (DEC-014).
- Tokenization: regex `[A-Za-z_][A-Za-z0-9_]*|\S`, giữ case (lowercase=False),
  không xử lý comment, số và punctuation thành token riêng.

## 9.5 TF-IDF Configuration
- max_features 20000 → vocab thực tế **11.495** (dataset không đủ token distinct).
- Doc length train: mean 625, median 524, max 15.005 chars.
- Sparsity ma trận: 0.9965. Cosine = dot product trên rows L2-normalized,
  batch 4096, toàn trình sparse.

## 9.6 Threshold Protocol
- Grid 0.01–0.99 trên **validation**, tối ưu F1 → **0.24** (val F1 0.6787).
- Freeze → đánh giá test đúng 1 lần. Test labels không dùng khi chọn ngưỡng.

## 9.7 Results

| Split | Prec. | Recall | F1 | ROC-AUC | PR-AUC | Acc |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 0.536 | 0.846 | 0.656 | 0.654 | 0.671 | 0.556 |
| Validation | 0.563 | 0.854 | 0.679 | 0.700 | 0.720 | 0.596 |
| Test | **0.552** | **0.860** | **0.673** | **0.690** | **0.704** | 0.581 |

Test ≈ validation (F1 chênh 0.006) → ngưỡng generalize tốt, không overfit
threshold selection.

## 9.8 Confusion Matrix (test, n=11.000)
TP 4.730 · TN 1.666 · FP 3.834 · FN 770.
Lỗi áp đảo là FP (gấp ~5 lần FN): baseline lexical thiên về "thấy giống từ là
bảo clone" ở ngưỡng F1-tối ưu thấp (0.24).

## 9.9 Error Analysis
- FP (3.834): cặp khác problem nhưng chia sẻ từ vựng/boilerplate C
  (`int main`, `#include`, vòng lặp IO) → cosine cao giả.
- FN (770): cặp cùng problem nhưng khác cách triển khai/từ vựng → cosine thấp.
- Chi tiết top-100 FP (score cao nhất) / FN (score thấp nhất):
  `results/exp001_tfidf_poj104/error_cases.jsonl` (lưu IDs + scores, truy code
  qua `data/processed/poj104/samples.jsonl`). Phân tích pattern sâu (độ dài,
  boilerplate) để phase error-analysis tổng hợp sau M1–M5.

## 9.10 Runtime (CPU-only)
fit 10.1s · transform 21.9s · scoring 6.5s · **total 44.6s** ·
peak Python alloc ~88.5MB. Không bottleneck — M1–M5 mới là bài toán hiệu năng.

## 9.11 Reproducibility
```powershell
.venv\Scripts\activate
python -m src.experiments.exp001_tfidf --config configs/experiments/exp001.yaml
```
Seed 42, manifest + config hash trong `run_metadata.json`, predictions
`predictions_test_scores.jsonl` (11k dòng) cho phép tính lại mọi metric.

## 9.12 Threats / Limitations
- Lexical: nhạy từ vựng/identifier, mù với tương đương ngữ nghĩa khác từ.
- Ngưỡng 0.24 thấp, phụ thuộc phân bố val; đổi val split có thể đổi ngưỡng.
- Chỉ classification; retrieval metrics chưa áp dụng (pair artifact hiện tại
  là classification — đúng §13, không ép).
- Hạn chế của dataset: dedup/normalization bảo thủ, near-dup chưa xử lý.

## 9.13 Conclusion
Kết quả thiết lập mốc lexical cho RQ1: test **P 0.552 / R 0.860 / F1 0.673**,
lỗi chủ yếu là FP do chia sẻ từ vựng xuyên problem. Mọi model M1–M5 phải vượt
mốc này trên cùng split/pairs/metrics mới chứng minh được giá trị của
representation mạnh hơn. Không kết luận gì về RQ1 ở đây ngoài mốc baseline.
