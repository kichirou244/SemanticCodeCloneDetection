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

Notebook Kaggle: `notebooks/exp002_codebert_kaggle.ipynb`, `notebooks/exp003_graphcodebert_kaggle.ipynb`
(bản EXP-003 là bản sao EXP-002 chỉ đổi tên — DEC-016; test `test_notebooks_mirror_each_other` chặn lệch).

**Chỉ tải checkpoint bản nhẹ về (0,5GB thay vì 1,44GB).** Sau khi train trên Kaggle,
gọi `export_eval_weights(Path("results/<exp>_poj104"))` để tạo `best_weights_only.pt`
(chỉ `model_state`, load được với `weights_only=True`); `evaluate_model` tự fallback
sang file này nếu không có checkpoint đầy đủ. `best_checkpoint.pt`/`last_checkpoint.pt`
chỉ dùng để resume trên Kaggle — đừng tải về (DEC-022 §2b).

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
