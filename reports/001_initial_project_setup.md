# 001 — Báo cáo tổng kết INIT

## Experiment
INIT-001 (initial project setup, không phải training experiment)

## Date
2026-09-20

## Repository state
- Trước INIT: repo trống — chỉ có `.agents/` (framework skill generic),
  `docs/` (templates + specs system), `README.md` stub 2 dòng.
- Không có src/configs/scripts/tests/data/results/reports/requirements.

## Đã tạo ở INIT
- Khung `src/` (common, core interfaces, data loaders/validators/dedup/
  splitting/leakage, representations ast/cpg, models B1 chạy được + M1–M5
  smoke/stub, evaluation, transformations registry, analysis).
- `configs/` (2 dataset + 5 model YAML), `scripts/` (6 script),
  `tests/` (5 test), `requirements.txt`, `pyproject.toml`, `.gitignore`.
- 10 docs nghiên cứu, decision log (6 DEC), registry (12 EXP).
- Reports 001–005 (file này + 4 file audit/feasibility).

## Dataset state
- Chưa tải dữ liệu thật (agent không chạy terminal). Loaders + audit scripts
  sẵn sàng; chờ người dùng chạy lệnh ở `docs/implementation_plan.md` §2.

## Architecture
Xem `project_structure.md`. Interface khóa: `DatasetAdapter`,
`CodeRepresentation`, `CloneDetector`, `Experiment`.

## Decisions
DEC-001–006 trong `docs/decision_log.md`.

## Blockers / Risks
1. Chưa xác thực schema thật của 2 dataset (reports 002/003 còn pending).
2. Joern CLI, tree-sitter grammars chưa kiểm tra (pending user run).
3. **Môi trường CPU-only** — rủi ro tốc độ cho M1/M2/M4/M5; cần quyết định
   GPU ở phase sau (ghi DEC mới).
4. M3/M4/M5 là stub — cần feasibility trước khi implement.

## Environment (kết quả thật, người dùng chạy 2026-09-20)
- Python 3.12.10, Windows 11, torch 2.14.0+cpu (**CPU-only, không CUDA**),
  transformers 5.17.0.
- Hệ quả: train/fine-tune Transformer full trên CPU sẽ rất chậm → phase sau
  chỉ train subset nhỏ local, hoặc cần GPU (Colab/Kaggle/máy lab). Ghi nhận
  làm constraint cho paper (hardware section).

## Smoke-test results
- `pytest tests/ -q`: **6 passed** (1 warning `token_pattern` — đã sửa bằng
  `token_pattern=None` trong `src/models/baseline/tfidf.py`).
- `scripts/*.py` chạy trực tiếp lỗi `ModuleNotFoundError: No module named
  'src'` (Python chỉ thêm `scripts/` vào path) — đã sửa bằng bootstrap
  `sys.path` trong 5 script. `env_report.py` không import `src` nên không ảnh
  hưởng.
- Chưa chạy lại sau sửa: `smoke_baseline`, `check_toolchain`,
  `audit_dataset` (x2), `check_leakage`, `smoke_transformer` (x2).

## Phát hiện chính (vòng chạy 2026-09-20)
1. Schema POJ-104 chốt: keys `[id, code, label]`, 53k mẫu / 104 problems.
2. **Official split POJ-104 FAIL leakage** (500 mẫu trùng xuyên split) →
   bắt buộc problem split (DEC-007). Finding đưa được vào paper.
3. Grammars tree-sitter cpp/python OK; Joern chưa cài (DEC-009).
4. CodeNet load full crash trên Windows (WinError 1450) → chuyển streaming
   (DEC-008); loader + scripts đã hỗ trợ `--split/--streaming`.
5. **CodeNet problem split vẫn rò rỉ code xuyên split (3/20k mẫu)** → dedup
   cross-problem trước split (DEC-010).
6. Schema CodeNet chốt: 11 keys gồm `status` (lọc accepted) + metadata;
   long-tail (median 8, 40% problems <5 mẫu); normalized dup ~3%.
7. Decision log lên 1.3.0 (DEC-007→DEC-010).

## Trạng thái INIT: HOÀN TẤT
Đủ 19 outputs: audit repo/.agents/docs, kiến trúc, env (CPU-only),
audit POJ-104 full + CodeNet 20k streaming, leakage strategy + bằng chứng,
loaders, interfaces, configs, B1 smoke đạt, M1/M2 smoke đạt, AST/CPG
feasibility (grammars OK, Joern pending), docs, reports, decision log,
registry, implementation plan, risks/blockers.

## Next actions (phase Pair Generation + EXP-001)
1. Pair generation POJ-104 trên problem split: dedup, stratify (problems
   65/81 gấp đôi), cân bằng clone/non-clone, seed cố định.
2. Train/eval B1 TF-IDF full POJ-104 → report 005-baseline (EXP-001).
3. Đo phân bố `status` CodeNet → DEC lọc accepted-only.
