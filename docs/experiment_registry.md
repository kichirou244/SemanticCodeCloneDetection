<a id="top"></a>

# Experiment registry (experiment_registry.md)

| Mục | Nội dung |
| :--- | :--- |
| **Phiên bản** | 1.4.0 |
| **Ngày cập nhật** | 2026-09-20 |
| **Tác giả** | Nhóm nghiên cứu |
| **Trạng thái** | Draft |

---

## Mục lục

- [1. Registry](#1-registry)
- [2. Lịch sử thay đổi](#2-lich-su-thay-doi)

---

## 1. Registry

| Experiment | Dataset | Model | Status | Result | Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EXP-001 | POJ-104 | B1 TF-IDF | COMPLETED | Test P/R/F1 0.552/0.860/0.673, thr 0.24 | reports/009_baseline_results.md |
| EXP-002 | POJ-104 | M1 CodeBERT | READY_FOR_GPU | impl + tests xong, chờ user chạy preflight/train | - |
| EXP-003 | POJ-104 | M2 GraphCodeBERT | PLANNED | - | - |
| EXP-004 | POJ-104 | M3 ASTNN | PLANNED | - | - |
| EXP-005 | POJ-104 | M4 CPG-GNN | PLANNED | - | - |
| EXP-006 | POJ-104 | M5 Hybrid | PLANNED | - | - |
| EXP-007 | CodeNet Python | B1 TF-IDF | PLANNED | - | - |
| EXP-008 | CodeNet Python | M1 CodeBERT | PLANNED | - | - |
| EXP-009 | CodeNet Python | M2 GraphCodeBERT | PLANNED | - | - |
| EXP-010 | CodeNet Python | M3 ASTNN | PLANNED | - | - |
| EXP-011 | CodeNet Python | M4 CPG-GNN | PLANNED | - | - |
| EXP-012 | CodeNet Python | M5 Hybrid | PLANNED | - | - |

Trạng thái: PLANNED / RUNNING / COMPLETED / FAILED / BLOCKED / INVALIDATED.

Ghi chú Phase 1 (2026-09-20): pipeline dataset chung đã implement
(`scripts/build_dataset.py` + manifests); config EXP-001 sẵn ở
`configs/experiments/exp001.yaml`. **POJ-104 pairs built + leakage passed
(353k pairs, manifest `data/manifests/poj104.json`) — sẵn sàng EXP-001.
CodeNet pilot finalized (547k pairs, accepted-allowlist, passed).**
EXP-001/EXP-007 vẫn PLANNED cho tới khi chạy model thật.

[🔝 Quay lại mục lục](#top)

---

## 2. Lịch sử thay đổi

| No. | Phiên bản | Ngày thay đổi | Nội dung thay đổi |
| :--- | :--- | :--- | :--- |
| 1 | 1.0.0 | 2026-09-20 | Khởi tạo 12 experiment core (6 POJ-104 + 6 CodeNet Python). |
| 2 | 1.1.0 | 2026-09-20 | Ghi chú Phase 1: pipeline dataset + config EXP-001 sẵn, status giữ PLANNED. |
| 3 | 1.2.0 | 2026-09-20 | POJ-104 pairs built + passed (sẵn sàng EXP-001); CodeNet pilot chờ rebuild. |
| 4 | 1.3.0 | 2026-09-20 | EXP-001 COMPLETED (test F1 0.673); report 009. |
| 5 | 1.4.0 | 2026-09-20 | EXP-002 READY_FOR_GPU (config + runner + tests xong). |
