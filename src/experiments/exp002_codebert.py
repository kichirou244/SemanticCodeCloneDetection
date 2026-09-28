"""EXP-002 — M1 CodeBERT trên POJ-104 (thin CLI wrapper).

Toàn bộ implementation nằm ở `transformer_pair_runner` dùng chung với
EXP-003 (M2 GraphCodeBERT) — cùng model size, cùng hyperparameter, cùng data
nên phải dùng chung một code path, nếu không RQ1 so sánh M1 vs M2 sẽ lẫn
implementation gap vào kết quả (DEC-016).

Module này giữ nguyên public API cũ để test và notebook cũ không đổi.

  python -m src.experiments.exp002_codebert --config configs/exp002_codebert.yaml --mode train
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
    export_eval_weights,
    load_code_map,
    load_eval_weights,
    load_pair_refs,
    load_training_state,
    main,
    preflight,
    resolve_eval_checkpoint,
    run_batches,
    save_training_state,
    train_model,
    truncation_analysis,
    write_history,
    write_run_metadata,
)

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
    "export_eval_weights",
    "load_code_map",
    "load_eval_weights",
    "load_pair_refs",
    "main",
    "preflight",
    "resolve_eval_checkpoint",
    "run_batches",
    "save_training_state",
    "train_model",
    "truncation_analysis",
    "write_history",
    "write_run_metadata",
]


if __name__ == "__main__":
    main()
