# 003 — Dataset audit CodeNet Python (PARTIAL — crash load full, chuyển streaming)

## Dataset
CodeNet Python — `windchimeran/codenet_python` (Python, core duy nhất)

## Language
Python

## Configuration
- Loader: `src/data/loaders/codenet_python.py` (đã thêm
  `split/streaming/max_samples/shuffle_buffer`, DEC-008)
- Smoke: `--max-samples 200` đạt (missing 0, normalized-dup 1/200).
- Load full **crash**: `datasets` build arrow cho split `big` lỗi
  `OSError Errno 22` khi ghi cache + `WinError 1450` khi dọn dẹp trên Windows
  (máy user, 2026-09-20). Nghi hết tài nguyên ổ C / giới hạn file Windows.

## Environment
Xem `reports/001_initial_project_setup.md` (CPU-only, không CUDA).

## Results (streaming 20.000 mẫu, split small, shuffle buffer 10k, 2026-09-20)
- **Schema thật: keys `[submission_id, problem_id, language, status, code_size,
  cpu_time, memory, code, input, output, problem_description]`** → loader đúng
  (`code`, `problem_id`), missing_code = 0.
- Có field `status` (accepted/rejected) + metadata hiệu năng + problem
  description — cơ sở để lọc accepted-only ở phase sau.
- 782 problems, phân bố long-tail: mean 25.6, median 8, max 740 (p00001);
  **311/782 problems có <5 mẫu (~40%), 139 singleton** → pair sampling phải
  đặt ngưỡng min-samples/problem + stratify.
- Duplicate: exact 2, **normalized 593/20000 (~3%)** — ủng hộ DEC-010.
- Full report JSON: `data/artifacts/codenet_small_20k_audit.json`.

## Observations
1. Streaming + shuffle buffer hoạt động tốt, không crash — xác nhận DEC-008.
2. Long-tail problem distribution là đặc trưng quan trọng của CodeNet cho paper.
3. HF cache đã sang ổ D (warning hiện `D:\.cache\...`) — kiểm tra lại biến
   HF_HOME nên là `D:\huggingface_cache` cho gọn (hiện cache nằm thẳng dưới D:\).

## Observations
1. Quy mô gấp ~84 lần POJ-104 — pipeline phải thiết kế cho sampling, không
   ôm full vào RAM.
2. Crash là vấn đề môi trường Windows + cache ổ C, không phải lỗi loader.
   Ổ C còn 26GB trống (kiểm tra 2026-09-20) → loại trừ hết dung lượng;
   nguyên nhân là giới hạn Windows khi build file arrow lớn.

## Limitations
- Chưa audit đại diện (200 mẫu đầu bias theo problem `p00001`).
- Streaming sample là ước lượng, không thay audit full.

## Next Actions
1. Đo phân bố `status` (accepted vs rejected) → quyết định lọc accepted-only.
2. Dedup cross-problem trước split (DEC-010) ở phase pair generation.
3. Ngưỡng min-samples/problem cho sampling (tail 40% problems <5 mẫu).
4. Audit full để phase sau (máy khỏe hơn).
