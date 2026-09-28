# Bản đồ cấu trúc project (project_structure.md)

> Tệp này ghi nhận phân bổ file/thư mục mã nguồn thực tế. Cập nhật mỗi khi thêm module mới.

## Cấu trúc tổng thể

```text
SemanticCodeCloneDetection/
├── .agents/                  # Quy chuẩn agent (có sẵn, không sửa)
├── configs/
│   ├── datasets/             # poj104.yaml, codenet_python.yaml
│   └── models/               # tfidf, codebert, graphcodebert, cpg_gnn, hybrid
├── data/                     # raw/ processed/ splits/ artifacts/ (không commit dữ liệu lớn)
├── src/
│   ├── common/               # seed.py, config.py
│   ├── core/                 # interfaces.py (DatasetAdapter, CloneDetector, ...)
│   ├── data/
│   │   ├── loaders/          # base.py, poj104.py, codenet_python.py
│   ├── cleaning/         # cleaner.py (DatasetCleaner, reasons)
│   ├── normalization/    # normalizer.py (CodeNormalizer, giữ raw)
│   ├── pairs/            # generator.py (PairGenerator 1:1, caps)
│   │   ├── validators/       # schema.py
│   │   ├── deduplication/    # duplicates.py
│   │   ├── splitting/        # problem_split.py
│   │   └── leakage/          # checks.py
│   ├── representations/
│   │   ├── ast/              # extract.py (tree-sitter) + store.py (ASTTreeStore)
│   │   └── cpg/              # schema.py (vocabulary CPG) · ingest.py (JSON→CPGraph)
│   │                          # graph.py (CPGraph, validate, collate PyG) · store.py
│   │                          # (CPGStore, cache .npz) · build.py (trạng thái toolchain)
│   ├── models/
│   │   ├── codebert_pair.py  # M1 siamese classifier [hA;hB;|hA-hB|]+MLP
│   │   ├── baseline/         # tfidf.py (B1, đã chạy được)
│   │   ├── transformer/      # embedder.py (M1/M2, smoke test)
│   │   ├── astnn/            # model.py (M3, vectorize RvNN + bi-GRU)
│   │   ├── cpg_gnn/          # model.py (M4, GAT 3 tầp có điều kiện nhãn cạnh)
│   │   └── hybrid/           # model.py (M5, stub, fusion concat_mlp)
│   ├── evaluation/           # metrics.py (classification + retrieval)
│   ├── experiments/          # exp001_tfidf.py (B1), exp002_codebert.py (M1)
│   ├── training/             # (phase sau)
│   ├── transformations/      # registry.py (RQ4, engine ở phase sau)
│   ├── analysis/             # errors.py (FP/FN)
│   └── utils/
├── scripts/                  # audit_dataset, check_leakage, build_dataset, audit_status,
│                              # smoke_*, env_report, check_toolchain,
│                              # build_cpg_export (lô gọi Joern + benchmark)
│   └── joern/                # export_cpg.sc (Scala, chạy bởi JVM) + joern_smoke.sc
├── notebooks/                # exp002/003/004/005 Kaggle (mỗi thí nghiệm 1 file)
├── tests/                    # test_baseline, test_splitting, test_metrics, test_leakage,
│                              # test_configs, test_cleaning, test_dedup_groups,
│                              # test_pairs, test_pair_leakage, test_exp00{1..4},
│                              # test_cpg_gnn, test_cpg_scripts
├── results/                  # output experiment (không commit checkpoints nặng)
├── reports/                  # research reports paper-ready
├── docs/                     # tài liệu nghiên cứu + specs hệ thống có sẵn
├── requirements.txt / pyproject.toml / .gitignore
└── README.md
```

## Lịch sử cập nhật

| Ngày | Nội dung |
| :--- | :--- |
| 2026-09-20 | Khởi tạo INIT: toàn bộ khung trên |
| 2026-09-28 | M4 (CPG-GNN): thêm `representations/cpg/{schema,ingest,graph,store}.py`,
`models/cpg_gnn/model.py` (thay stub), `scripts/build_cpg_export.py`,
`scripts/joern/{export_cpg.sc,joern_smoke.sc}`, `configs/exp005_cpg_gnn.yaml`,
`notebooks/exp005_cpg_extract_kaggle.ipynb`, `tests/test_cpg_{gnn,scripts}.py` |
