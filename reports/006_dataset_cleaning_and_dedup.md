# 006 — Dataset cleaning & dedup (POJ-104 DONE — CodeNet chờ rebuild)

## Phase
Phase 1 — Dataset Protocol & Pair Generation (Steps 5–8)

## Date
2026-09-20

## Dataset
POJ-104 + CodeNet Python

## Language
C/C++ + Python

## Configuration
- Cleaner: `src/data/cleaning/cleaner.py` (min_code_chars 10,
  expected_language, status_allowlist).
- Normalizer: `src/data/normalization/normalizer.py` (giữ raw).
- Dedup: `src/data/deduplication/groups.py` (canonical = sample_id nhỏ nhất).
- Configs: `configs/pairs/poj104.yaml`, `configs/pairs/codenet_python.yaml`.

## Input size
- POJ-104: 53.000 raw → cleaning giữ 53.000 (loại 0, dataset sạch).
- CodeNet pilot: 200.000 streaming split small.

## Removed samples / Duplicate statistics (POJ-104, từ manifest thật)
- Dedup: input 53.000 → canonical 51.991, loại 1.009
  (1.005 exact groups + 4 normalized groups).
- Ineligible problems: 0 (mọi problem ≥ 2 mẫu).

## Removed samples / Duplicate statistics (CodeNet pilot 200k small)
- Cleaning: giữ 200.000, loại 0.
- Dedup: input 200.000 → canonical 194.083, loại 5.917
  (154 exact groups + 4.705 normalized groups) — cross-problem dups đúng như DEC-010.
- Ineligible: 590 problems <5 mẫu, loại có reason đầy đủ.
- Rebuild với `status_allowlist=["Accepted"]` (2026-09-20): số **không đổi**
  (cleaning vẫn loại 0) — xác nhận pilot small thuần Accepted, allowlist chỉ
  có tác dụng bảo vệ khi mở rộng sang big/full.

## Decisions
- DEC-011 (pair ratio/caps), DEC-013 (POJ-104 dup policy).
- DEC-012 (accepted-only): small 100% Accepted nhưng big chỉ ~60% →
  enforce `status_allowlist=["Accepted"]` mọi build CodeNet; loader đã đưa
  `status` vào metadata.

## Limitations
- Normalization bảo thủ (không xử lý comment) — near-dup qua comment khác
  nhau chưa bắt được; phase sau.
- CodeNet pilot 200k, chưa full 4.4M; manifest CodeNet chờ rebuild với allowlist.

## Next Actions
1. Rebuild CodeNet pilot với allowlist → lấy manifest số thật.
2. Sang EXP-001 với POJ-104 pairs.
