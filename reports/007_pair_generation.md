# 007 — Pair generation (POJ-104 DONE — CodeNet chờ rebuild)

## Phase
Phase 1 — Steps 12–17

## Date
2026-09-20

## Pair generation strategy
- Positive = cùng problem (sau dedup); negative = khác problem, 1:1 (DEC-011).
- Cap positive/problem: POJ-104 train 2000 / val-test 500; CodeNet pilot
  train 500 / val-test 200.
- Pair ID vô thứ tự; deterministic theo seed/split; không oversample test.
- Implement: `src/data/pairs/generator.py`; build: `scripts/build_dataset.py`.

## Positive/Negative count (POJ-104, từ manifest thật)
- train: 166.000 pos + 166.000 neg (83 problems × cap 2000 — cap có tác dụng,
  đúng stratification).
- validation: 5.000 + 5.000 (10 × 500). test: 5.500 + 5.500 (11 × 500).
- Tổng: 353.000 pairs, tỉ lệ 1:1 mọi split.

## Split statistics (POJ-104)
- train: 41.491 samples / 83 problems; validation: 5.000 / 10;
  test: 5.500 / 11 (tổng 104 problems, không overlap).
- Artifacts: `data/splits/poj104/*_problems.json`,
  `data/pairs/poj104/{train,validation,test}.jsonl`,
  `data/processed/poj104/samples.jsonl`, `data/manifests/poj104.json`.

## Positive/Negative count (CodeNet pilot, từ manifest thật)
- train: 242.146 + 242.146 (830 problems, cap 500).
- validation: 15.837 + 15.837 (103 problems, cap 200).
- test: 15.763 + 15.763 (105 problems, cap 200).
- Tổng ~547k pairs, 1:1 mọi split. Artifacts tương ứng dưới
  `data/{splits,pairs,processed,manifests}/codenet_python*`.

## Rationale & impact (chống tối ưu metric)
- Cap làm giảm số pair problems lớn nhưng giữ phân bố problem (stratify),
  không loại samples khó; test phản ánh đúng protocol định sẵn.
- Mọi thay đổi protocol sau khi xem metric → invalidate experiment (§29).

## Limitations
- Negative cross-problem có thể lẫn accidental semantic equivalence (nhiễu
  chấp nhận được, ghi nhận cho Threats).
- Hard negative để phase sau.

## Next Actions
1. Rebuild CodeNet pilot với allowlist → manifest số thật.
2. EXP-001 đọc pairs POJ-104 từ `data/pairs/poj104/` + code từ processed.
