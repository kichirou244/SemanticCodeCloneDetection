# Semantic Code Clone Detection (C/C++)

Project nghiên cứu phục vụ bài báo khoa học: xây dựng và đánh giá các phương pháp
AI/Deep Learning phát hiện hai đoạn code tương đương ngữ nghĩa/chức năng.

## Phạm vi core (không mở rộng nếu chưa có yêu cầu rõ ràng)

| ID | Phương pháp | Nhóm |
| :--- | :--- | :--- |
| B1 | TF-IDF + Cosine | Baseline |
| M1 | CodeBERT | Transformer |
| M2 | GraphCodeBERT | Transformer |
| M3 | ASTNN | AST |
| M4 | CPG-GNN (backbone `gat`) | Program Graph |
| M5 | GraphCodeBERT + CPG-GNN (`concat_mlp`) | Hybrid |

- Dataset duy nhất trong phạm vi: **POJ-104 (C/C++)**
  (`google/code_x_glue_cc_clone_detection_poj104`) → EXP-001…EXP-006
- **Ngôn ngữ: C/C++.** Khối Python (CodeNet) đã hủy theo DEC-019 vì quota
  compute; dữ liệu CodeNet vẫn giữ trong repo để mở rộng lại được nếu cần.
  Xem `docs/research_questions.md` §5.
- Không Java/C#/JavaScript.
- **M4 (CPG-GNN):** code đã xong (schema + ingest + store + GAT, 34 test local),
  còn thiếu **CPG thật** vì cần Joern. Joern **không cài ở máy local** — cài trong
  `notebooks/exp005_cpg_extract_kaggle.ipynb` rồi chạy **cell smoke trước** (in ra
  danh sách step CPGQL để xác nhận exporter), sau đó mới đo `ham/giây` thật.
  Xem DEC-024, DEC-025 và `reports/005_ast_cpg_feasibility.md`.

## Bắt đầu nhanh (người dùng tự chạy — agent không chạy terminal)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/env_report.py
pytest tests/ -q
python scripts/smoke_baseline.py
python scripts/audit_dataset.py --dataset poj104 --max-samples 200
python scripts/check_leakage.py --dataset poj104
python scripts/check_toolchain.py
```

Các lệnh nặng (tải dataset đầy đủ, smoke transformer) xem chi tiết trong
`docs/implementation_plan.md` và `docs/reproducibility.md`.

## Chạy thí nghiệm (EXP-001…EXP-004)

Mọi mode đều dùng **cùng một CLI**; chỉ đổi `--config` và `--mode`.

```bash
# EXP-002 / EXP-003 — cùng code path, chỉ khác pretrained weights
python -m src.experiments.exp002_codebert      --config configs/exp002_codebert.yaml      --mode preflight --device cuda
python -m src.experiments.exp003_graphcodebert --config configs/exp003_graphcodebert.yaml --mode diagnostic --device cuda

# EXP-004 — ASTNN (không cần tokenizer, tự build cache AST)
python -m src.experiments.exp004_astnn --config configs/exp004_astnn.yaml --mode preflight --device cuda
python -m src.experiments.exp004_astnn --config configs/exp004_astnn.yaml --mode diagnostic --device cuda
```

`--mode`: `preflight` → `smoke` → `diagnostic` → `train` → `evaluate`, cộng
`resume` nếu train bị ngắt. **Local chạy `smoke`/`diagnostic`/`evaluate`**; `train`
để Kaggle theo DEC-018 (checkpoint nằm trong `/kaggle/working` ⇒ phải archive
ngay cuối mỗi epoch).

Notebook Kaggle (mỗi thí nghiệm một file, tự build dataset, tự đóng gói):

| Notebook | Dùng cho | Thời gian T4 |
| :--- | :--- | ---: |
| `notebooks/exp002_codebert_kaggle.ipynb` | EXP-002 (M1) — đã xong | 7,03h (đo) |
| `notebooks/exp003_graphcodebert_kaggle.ipynb` | EXP-003 (M2) | ~7,8h |
| `notebooks/exp004_astnn_kaggle.ipynb` | EXP-004 (M3) | ~35 phút |

Chạy **EXP-004 trước EXP-003** (DEC-023): M3 rẻ hơn 13×, nếu phiên M2 chết giữa chừng thì
vẫn còn M3. Notebook EXP-003 là bản sao EXP-002 chỉ đổi tên — DEC-016; test
`test_notebooks_mirror_each_other` chặn lệch.

**Mỗi notebook có cell evaluate chạy ngay trên Kaggle** (M2 ~5 phút, M3 ~1 phút), rồi đóng
gói **chỉ metrics + predictions** (vài MB) — mặc định `KEEP_WEIGHTS = False` nên không
tải checkpoint (1,44GB) về máy. Muốn giữ `best_weights_only.pt` (0,5GB) để chạy lại ở máy
khác thì bật cờ đó. Mọi metric đều tính lại được từ `predictions/*_scores.jsonl` mà không
cần model.

**Chỉ tải checkpoint bản nhẹ về khi thật cần (0,5GB thay vì 1,44GB).** Sau khi train trên
Kaggle, gọi `export_eval_weights(Path("results/<exp>_poj104"))` để tạo `best_weights_only.pt`
(chỉ `model_state`, load được với `weights_only=True`); `evaluate_model` tự fallback
sang file này nếu không có checkpoint đầy đủ. `best_checkpoint.pt`/`last_checkpoint.pt`
chỉ dùng để resume trên Kaggle (DEC-022 §2b, DEC-023).

Thời gian trên Kaggle T4: **EXP-002 7,03h (đo)** · EXP-003 ~7,8h (đo từ throughput
EXP-002) · **EXP-004 ~35 phút**. Xem `docs/experiment_registry.md` cho trạng thái từng run.

## Tài liệu

- Tổng quan: `docs/project_overview.md`
- Câu hỏi nghiên cứu: `docs/research_questions.md`
- Dataset/split/leakage: `docs/dataset_protocol.md`, `docs/leakage_prevention.md`
- Experiment/đánh giá: `docs/experiment_protocol.md`, `docs/evaluation_protocol.md`
- Quyết định: `docs/decision_log.md` · Registry: `docs/experiment_registry.md`
- Kế hoạch: `docs/implementation_plan.md` · Reports: `reports/`

## Nguyên tắc

Code + Documentation + Research Reports phát triển song song. Mọi thay đổi về
dataset/model/split/metric/leakage/kết quả/quyết định phải cập nhật docs/report
tương ứng. Chi tiết: `docs/reproducibility.md`.
