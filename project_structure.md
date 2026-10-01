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
│                              # library/ = database quét clone (file .c thả vào là quét được)
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
│   │   ├── baseline/         # tfidf.py (EXP-001, lịch sử) · moss.py (EXP-006,
│   │   │                     #   winnowing MOSS local — baseline demo + scan)
│   │   ├── transformer/      # embedder.py (M1/M2, smoke test)
│   │   ├── astnn/            # model.py (M3, vectorize RvNN + bi-GRU)
│   │   ├── cpg_gnn/          # model.py (M4, GAT 3 tầp có điều kiện nhãn cạnh)
│   │   └── hybrid/           # model.py (M5, stub, fusion concat_mlp)
│   ├── evaluation/           # metrics.py (classification + retrieval)
│   ├── experiments/          # exp001_tfidf.py (B1 cũ), exp002_codebert.py (M1),
│   │                          # exp006_moss.py (B1 MOSS baseline mới)
│   ├── training/             # (phase sau)
│   ├── transformations/      # registry.py (RQ4, engine ở phase sau)
│   ├── analysis/             # errors.py (FP/FN)
│   ├── demo/                 # web demo (M6 · Dùng 4 model đã train chấm điểm cặp code)
│   │   ├── alignment/        # lexer.py (C++ per-line) + aligner.py (LCS → khối khớp)
│   │   ├── scorers/          # base.py (giao diện) + moss/transformer/astnn_scorer.py
│   │   ├── registry.py       # nạp model từ configs/demo_web.yaml (B1,M1,M2,M3)
│   │   ├── service.py        # DemoService: analyze/status/real_pairs/library scan/_validate
│   │   ├── schemas.py        # DTO (AnalyzeRequest, PreloadRequest)
│   │   ├── server.py         # FastAPI: tĩnh + API, controller mỏng
│   │   ├── samples.py        # 5 mẫu tay (Type-1/2/3, near-miss, unrelated) + POJ-104
│   │   └── web/              # index.html · css/{tokens,app,scores,diff,scan}.css
│   │                          # js/{main,api,components,state,utils} (ES modules, không CDN JS)
│   └── utils/
├── scripts/                  # audit_dataset, check_leakage, build_dataset, audit_status,
│                              # smoke_*, env_report, check_toolchain,
│                              # build_cpg_export (lô gọi Joern + benchmark), serve_demo (uvicorn)
│   └── joern/                # export_cpg.sc (Scala, chạy bởi JVM) + joern_smoke.sc
├── notebooks/                # exp002/003/004/005 Kaggle (mỗi thí nghiệm 1 file)
├── tests/                    # test_baseline, test_splitting, test_metrics, test_leakage,
│                              # test_configs, test_cleaning, test_dedup_groups,
│                              # test_pairs, test_pair_leakage, test_exp00{1..4},
│                              # test_cpg_gnn, test_cpg_scripts, test_demo_{lexer,aligner,server},
│                              # test_moss (winnowing core)
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
| 2026-09-28 | Web demo (M6): `src/demo/**` + `src/demo/web/**` (UI thuần, không CDN),
`scripts/serve_demo.py`, `configs/demo_web.yaml`, `tests/test_demo_{lexer,aligner,server}.py`,
thêm `fastapi`/`uvicorn`/`httpx` vào `requirements.txt` |
| 2026-09-28 | Baseline = MOSS: `src/models/baseline/moss.py` (winnowing local, k=50/w=100,
crc32 tất định), `src/experiments/exp006_moss.py` + `configs/exp006_moss_poj104.yaml`,
demo B1 đổi sang `scorers/moss_scorer.py` (bỏ TF-IDF khỏi demo, giữ `baseline/tfidf.py`
cho EXP-001 cũ), chế độ **quét thư mục** `data/library/`: `GET/POST /api/library*`
trong `service.py`/`server.py`, `web/js/components/scanPanel.js` + `web/css/scan.css`,
`tests/test_moss.py`, 5 file `.c` mẫu trong `data/library/` |
