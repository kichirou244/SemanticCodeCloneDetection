# 002 — Dataset audit POJ-104 (FULL — 53k mẫu, đã chốt schema)

## Dataset
POJ-104 — `google/code_x_glue_cc_clone_detection_poj104` (C/C++, core duy nhất)

## Language
C/C++

## Configuration
- Loader: `src/data/loaders/poj104.py`
- Lệnh: `python scripts/audit_dataset.py --dataset poj104` (full, không slice)
- Seed: 42, Python 3.12.10, Windows 11, torch CPU-only. Ngày chạy: 2026-09-20.

## Environment
Xem `reports/001_initial_project_setup.md` (CPU-only, không CUDA).

## Results (full 53.000 mẫu)
- **Schema thật: keys `[id, code, label]`** → loader đúng (`code` ← code,
  problem ← `label`), missing_code = 0.
- Split chính thức: train 32500 / validation 8500 / test 12000 (~13MB parquet).
- 104 problems: 102 problems × 500 mẫu + problems **65, 81 × 1000 mẫu**
  (mất cân bằng nhẹ, ghi nhận cho paper).
- Duplicate toàn dataset: exact 1005/53000 (~1.9%), normalized 1009.

## Observations
1. Loader dự kiến khớp schema thật ngay lần đầu — không cần sửa loader.
2. Hai problems gấp đôi số mẫu có thể làm lệch pair sampling nếu không stratify.
3. ~1.9% exact duplicate: phải quyết định giữ/loại trước train (xem Next Actions).

## Limitations
- Chưa điều tra problem ID nào gây overlap ở official split (sang report 004).
- Near-duplicate (ngưỡng tương tự, không exact) chưa đo.

## Next Actions
1. DEC giữ/loại 1005 exact duplicates ở phase pair generation.
2. Dùng problem split cho mọi core experiment (DEC-007).
