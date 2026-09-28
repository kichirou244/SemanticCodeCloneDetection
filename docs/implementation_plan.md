<a id="top"></a>

# Implementation plan (implementation_plan.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.5.0 |
| **Ngày cập nhật** | 2026-09-28 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Đã xong ở INIT](#1-da-xong-o-init)
- [2. Việc người dùng chạy kiểm chứng](#2-viec-nguoi-dung-chay-kiem-chung)
- [3. Phase tiếp theo](#3-phase-tiep-theo)
- [4. Lịch sử thay đổi](#4-lich-su-thay-doi)

---

## 1. Đã xong ở INIT

- [x] Audit `.agents/` + `docs/` + repository (trống, chỉ có framework).
- [x] Kiến trúc `src/`, `configs/`, `scripts/`, `tests/`, `reports/`, `docs/`.
- [x] `requirements.txt`, `pyproject.toml`, `.gitignore`, `project_structure.md`.
- [x] Loaders POJ-104 + CodeNet Python, audit, dedup, problem split, leakage checks.
- [x] Interface `DatasetAdapter/CodeRepresentation/CloneDetector/Experiment`.
- [x] B1 TF-IDF chạy được + smoke script + test.
- [x] M1/M2 embedder + smoke script (chưa chạy — cần mạng).
- [x] Stub M3/M4/M5 + AST/CPG abstraction + toolchain check.
- [x] Metrics classification/retrieval + error saver + transformation registry.
- [x] 10 docs nghiên cứu + decision log (6 DEC) + registry (12 EXP) + 5 reports.

Ghi chú: project này không thuộc phạm vi web/app nên checklist
`web_conversion_protocol` không áp dụng (theo development_protocol bước 2).

[🔝 Quay lại mục lục](#top)

---

## 2. Việc người dùng chạy kiểm chứng

Agent không được chạy terminal, nên người dùng tự chạy theo thứ tự:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/env_report.py
pytest tests/ -q
python scripts/smoke_baseline.py
python scripts/check_toolchain.py
python scripts/audit_dataset.py --dataset poj104 --max-samples 200
python scripts/audit_dataset.py --dataset codenet_python --max-samples 200
python scripts/check_leakage.py --dataset poj104
python scripts/smoke_transformer.py --method codebert
python scripts/smoke_transformer.py --method graphcodebert
```

Kết quả dán lại cho agent để chốt vào reports 002/003/004 và feasibility.

[🔝 Quay lại mục lục](#top)

---

## 3. Phase tiếp theo

1. ~~Chốt audit dataset thật~~ — xong (reports 002/003/004, DEC-007/008/010).
2. ~~Toolchain AST~~ (grammars OK) — Joern pending (DEC-009).
3. ~~Phase 1: build dataset thật~~ — **đã khép** (reports 006/007/008, cả POJ-104
   353k pairs và CodeNet pilot 547k pairs đều `leakage_passed: true`).
4. ~~EXP-001 (B1 TF-IDF)~~ — **COMPLETED**, test P/R/F1 0.552/0.860/0.673
   (reports/009). Đây là mốc mà M1–M5 phải vượt.
5. ~~EXP-002 (M1 CodeBERT)~~ — **COMPLETED**, test P/R/F1 **0.931/0.890/0.910**,
   ROC-AUC 0.973, ngưỡng 0.59 (chọn trên val) — vượt EXP-001 **+0.237 F1**
   (reports/011).
   - ~~Audit + fix F-01/F-02~~ (DEC-017, reports/010).
   - ~~Diagnostic GPU~~ PASS trên RTX 5060.
   - ~~Kaggle run protocol~~ (DEC-018): flush mỗi epoch, `--time-budget-s`,
     `--mode resume`, notebook đã sửa.
   - ~~Train trên Kaggle~~: 3/3 epoch, 7,03h, Tesla T4, best epoch 1
     (`device_actual: cuda:0` trong `run_metadata.json`).
   - **Còn lại:** ablation truncation (`max_length 512`) + `min_delta` cho
     early stopping — **sau** khi có EXP-003, không sửa recipe giữa chừng (DEC-022).
6. ~~EXP-003 (M2 GraphCodeBERT) — **implementation xong, chờ train Kaggle.**~~
   Không clone runner: đã tách `transformer_pair_runner` dùng chung với EXP-002
   (DEC-020) nên mọi khác biệt quy về pretrained weights. **~7,8h GPU** (đo).
7. ~~**Implement M3 (ASTNN)** — **xong** (DEC-021)~~: extractor thật
   (`src/representations/ast/extract.py`), RvNN vectorize (`src/models/astnn/`),
   `ASTTreeStore` + cache, runner + config + 25 test. Kèm EXP-004.
   **~35 phút GPU** (đo + hiệu chuẩn T4), không phải ~2h như ước lượng cũ.
8. **Implement M4 (CPG-GNN)** — cần Joern cho C++ (DEC-009), dùng
   `torch-geometric`. Kèm EXP-005. Chưa đo.
9. **Implement M5 (Hybrid `concat_mlp`)** — cần M2 + M4 hoàn chỉnh. Kèm EXP-006.
   **~8h GPU**
10. Transformation engine cho RQ4 (8 tên đã khóa trong
    `src/transformations/registry.py`) → ablation → error analysis → báo cáo.

**Tổng GPU còn lại: ~16h** (EXP-003 ~7,8h đã đo + EXP-004 ~0,6h đã đo + EXP-005
chưa đo + EXP-006 ~8h) → vừa quota 30h/tuần của Kaggle. Early stopping
(patience 2) có thể cắt ngắn thêm. Số `~2h` cho EXP-004/005 trước đây là
ước lượng suông, **đã bị thu hồi** — xem DEC-021.

**Ràng buộc vận hành:** local dùng cho smoke test (<5 phút) **và `--mode evaluate`**
(EXP-002 mất 5 m 16 s trên RTX 5060 — DEC-022). Mọi EXP cần >5 phút **train** đều chạy
trên Kaggle theo DEC-018.

[🔝 Quay lại mục lục](#top)

---

## 4. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo ở phase INIT. |
| 2 | 1.1.0 | 2026-09-20 | Cập nhật Phase 1: pipeline dataset + lệnh user chạy. |
| 3 | 1.2.0 | 2026-09-27 | Phase 1 đã khép; EXP-001 COMPLETED; EXP-002 sẵn sàng train trên Kaggle (DEC-018). |
| 4 | 1.3.0 | 2026-09-27 | DEC-019: scope chỉ C/C++ — lộ trình còn EXP-003…006, ~20h GPU. |
| 5 | 1.4.0 | 2026-09-27 | DEC-020/021: M2 tách runner dùng chung, M3 implement xong. Sửa estimate EXP-004 từ ~2h (suông) xuống ~35 phút (đo + hiệu chuẩn T4). |
| 6 | 1.5.0 | 2026-09-28 | EXP-002 → COMPLETED (test F1 0.910, +0.237 so với EXP-001; reports/011). Bổ sung DEC-022: đóng băng recipe M1/M2, evaluate chạy local (5 m 16 s) thay vì mở phiên Kaggle. |
