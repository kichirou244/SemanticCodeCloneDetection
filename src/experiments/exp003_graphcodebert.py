"""EXP-003 — M2 GraphCodeBERT trên POJ-104.

Tham chiếu: Ahmad et al. (2021), "GraphCodeBERT: Empowering Transformers with
Graph Structures in Code Understanding".

Bối cảnh quyết định (DEC-016, RQ1): M1 và M2 đều là encoder RoBERTa-base 125M
với vocab 50265, nên **chi phí train gần như bằng nhau**. GraphCodeBERT khác
CodeBERT ở chỗ có thêm data-flow bias trong pretraining, không phải ở kiến trúc
hạ tầng → đây là baseline công bằng để so sánh "transformer thuần" với
"transformer + graph signal".

Implementation dùng **chung** `transformer_pair_runner` với EXP-002: khác nhau
duy nhất là `configs/exp003_graphcodebert.yaml` trỏ tới
`microsoft/graphcodebert-base`. Nhờ đó mọi khác biệt về số liệu đều quy về
pretrained weights, không quy về code path.

  preflight:  python -m src.experiments.exp003_graphcodebert --config configs/exp003_graphcodebert.yaml --mode preflight --device cuda
  diagnostic: ... --mode diagnostic --device cuda
  smoke:      ... --mode smoke --device cuda
  train:      ... --mode train --device cuda [--time-budget-s 32400]
  evaluate:   ... --mode evaluate --device cuda
  resume:     ... --mode resume --device cuda
"""

from src.experiments.transformer_pair_runner import (  # noqa: F401
    DEFAULT_TIME_BUDGET_S,
    DIAG_TRAIN_PAIRS,
    DIAG_VAL_PAIRS,
    SMOKE_TRAIN_PAIRS,
    SMOKE_VAL_PAIRS,
    _atomic_write_text,
    assert_device_setup,
    collect_device_metadata,
    diagnostic,
    evaluate_model,
    load_code_map,
    load_pair_refs,
    load_training_state,
    main,
    preflight,
    run_batches,
    save_training_state,
    train_model,
    truncation_analysis,
    write_history,
    write_run_metadata,
)

# Re-export ĐÚNG danh sách của exp002_codebert, để hai wrapper thay thế được lẫn
# nhau (`from src.experiments.exp003_graphcodebert import train_model` hoạt động
# y hệt). Nếu danh sách lệch, test `test_runner_is_shared_code_path` đỏ.
__all__ = [
    "DEFAULT_TIME_BUDGET_S",
    "SMOKE_TRAIN_PAIRS",
    "SMOKE_VAL_PAIRS",
    "DIAG_TRAIN_PAIRS",
    "DIAG_VAL_PAIRS",
    "assert_device_setup",
    "collect_device_metadata",
    "diagnostic",
    "evaluate_model",
    "load_code_map",
    "load_pair_refs",
    "main",
    "preflight",
    "run_batches",
    "save_training_state",
    "train_model",
    "truncation_analysis",
    "write_history",
    "write_run_metadata",
]


if __name__ == "__main__":
    main()
