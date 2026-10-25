# Bản đồ cấu trúc project (project_structure.md)

> Tệp này ghi nhận phân bổ file/thư mục mã nguồn thực tế. Cập nhật mỗi khi thêm module mới.

## Cấu trúc tổng thể

```text
SemanticCodeCloneDetection/
├── .agents/                  # Quy chuẩn agent (có sẵn, không sửa)
├── configs/
│   ├── datasets/             # poj104.yaml, codenet_python.yaml
│   ├── models/               # tfidf, codebert, graphcodebert, cpg_gnn, hybrid
│   └── final_pipeline.yaml   # EXP-FINAL-001: candidate runs + luật chọn + model inference
├── data/                     # raw/ processed/ splits/ artifacts/ (không commit dữ liệu lớn;
│   │                         #   ngoại lệ `data/custom_suite/` — bộ test tự viết, nằm trong git)
│   └── library/              # database quét clone (file .c thả vào là quét được)
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
│   ├── evaluation/           # metrics.py (classification + retrieval) ·
│   │                          # uncertainty.py (bootstrap + McNemar) ·
│   │                          # leaderboard.py (xếp hạng + chọn phương pháp cuối)
│   ├── experiments/          # exp001_tfidf.py (B1 cũ), exp002_codebert.py (M1),
│   │                          # exp006_moss.py (B1 MOSS baseline mới)
│   ├── training/             # (phase sau)
│   ├── transformations/      # registry.py (RQ4, engine ở phase sau)
│   ├── analysis/             # errors.py (FP/FN)
│   ├── demo/                 # web demo (M6 · Dùng 4 model đã train chấm điểm cặp code)
│   │   ├── alignment/        # lexer.py (C++ per-line) + aligner.py (LCS → khối khớp)
│   │   ├── scorers/          # base.py (giao diện) + moss/transformer/astnn/cpg/hybrid_scorer.py
│   │   │                         # cpg_scorer.py giữ extract_graphs() dùng chung M4+M5
│   │   ├── registry.py       # nạp model từ configs/demo_web.yaml (B1,M1..M5)
│   │   ├── service.py        # DemoService: analyze/status/real_pairs/library scan/_validate
│   │   ├── schemas.py        # DTO (AnalyzeRequest, PreloadRequest)
│   │   ├── server.py         # FastAPI: tĩnh + API, controller mỏng
│   │   ├── samples.py        # 5 mẫu tay (Type-1/2/3, near-miss, unrelated) + POJ-104
│   │   └── web/              # index.html · css/{tokens,app,scores,diff,scan}.css
│   │                          # js/{main,api,components,state,utils} (ES modules, không CDN JS)
│   └── utils/
├── scripts/                  # audit_dataset, check_leakage, build_dataset, audit_status,
│                              # smoke_*, env_report, check_toolchain,
│                              # build_cpg_export (lô gọi Joern + benchmark), serve_demo (uvicorn),
│                              # final_pipeline (select/detect — pipeline cuối EXP-FINAL-001),
│                              # score_custom_suite + summarize_custom_suite +
│                              # build_custom_suite_negatives (bộ test 89 cặp)
│   └── joern/                # export_cpg.sc (Scala, chạy bởi JVM) + joern_smoke.sc
├── notebooks/                # exp002/003/004/005 Kaggle (mỗi thí nghiệm 1 file)
├── tests/                    # test_baseline, test_splitting, test_metrics, test_leakage,
│                              # test_configs, test_cleaning, test_dedup_groups,
│                              # test_pairs, test_pair_leakage, test_exp00{1..4},
│                              # test_cpg_gnn, test_cpg_scripts, test_demo_{lexer,aligner,server},
│                              # test_moss (winnowing core), test_custom_suite_manifest
├── results/                  # output experiment (không commit checkpoints nặng)
├── reports/                  # research reports paper-ready
│   ├── figures/              # prompt tạo hình (m2_pipeline_prompt.md) — ảnh sinh ra từ
│   │                          # prompt rồi chèn vào báo cáo
│   └── images/               # ảnh chèn vào báo cáo (Pipeline.png: kiến trúc suy luận M2)
├── docs/                     # tài liệu nghiên cứu + specs hệ thống có sẵn
├── requirements.txt / pyproject.toml / .gitignore
└── README.md
```

## Lịch sử cập nhật

| Ngày | Nội dung |
| :--- | :--- |
| 2026-10-25 | **Mở rộng non-clone 16 → 64 cặp, tổng 89 cặp (DEC-034):** thêm 48 cặp non-clone sinh
tất định bằng `scripts/build_custom_suite_negatives.py` (24 cùng bài khác lõi, 6 base–base còn
thiếu, 18 khác bài cùng biến đổi), giữ nguyên 41 cặp cũ. `scripts/score_custom_suite.py` thêm
`--only-new` + `--batch-pairs` (Joern một lần cho M4/M5); `scripts/summarize_custom_suite.py`
xuất mean ± SD + bootstrap ra `workspace/custom_suite_summary.{json,md,tex}`; thêm
`tests/test_custom_suite_manifest.py`. Kết quả 89 cặp: recall 1,000 cho cả 5 mô hình, F1 M2
0,694 / M1 0,685 / M5 0,658 / M3 0,500 / M4 0,446; B1 chỉ có 41 cặp gốc nên so trên 41 cặp
chung (M2 0,909 dẫn đầu, B1 0,400). Cập nhật `data/custom_suite/README.md`,
`docs/evaluation_protocol.md` §3.1 và báo cáo `reports/[24 - 10] - Báo cáo.tex` (9 trang) |
| 2026-10-24 | **Thay nhóm P6 126–148 dòng bằng nhóm P6 83–93 dòng (DEC-033):** `data/custom_suite`
đổi 6 tệp P6 sang bài danh sách liên kết chèn theo thứ tự (`p6_base_list`, `p6_v_class`,
`p6_v_while/scanf/combo`, non-clone `p6_n_filter` thay `p6_n_reverse`; khác biệt nằm ở giữa
tệp), 7 cặp giữ nguyên id, tổng vẫn 41. Chấm lại hai đợt (`B1 M1 M2 M3` cho 41 cặp; `M4 M5`
cho 7 cặp P6 vì Joern khởi động ~2 phút/cặp) rồi `--merge`: M1 6 lỗi, M2 5, M3 13, M4 16,
M5 7; B1 chạy winnowing cục bộ nên 19 lỗi. Cập nhật `workspace/token_lengths.json`,
`data/custom_suite/README.md`, báo cáo `reports/[24 - 10] - Báo cáo.tex` (viết ngắn, mục
kế hoạch để trống) |
| 2026-10-17 | **Bộ test mở rộng nhóm P5 — danh sách liên kết + hướng đối tượng (DEC-032, theo góp ý
của thầy):** `data/custom_suite` thêm 6 file C++ dài 59–70 dòng (`p5_base_list`: `struct Node`
+ 4 hàm tự do; `p5_v_class`: `class IntList` + 4 phương thức cùng hành vi; `p5_v_while/scanf/combo`;
`p5_n_palindrome`: cùng khung `class`/`Node` nhưng khác ngữ nghĩa) và 8 cặp mới (26 → 34 cặp).
`.gitignore` mở ngoại lệ `!data/custom_suite/` để code test vào git;
`scripts/score_custom_suite.py` thêm `--pairs`; `scripts/evidence_custom_suite.py` thêm `--models`
và tự dùng `\scriptsize` khi code dài (`scan_demo_zip.py::_latex_minipage` thêm tham số `font`);
báo cáo tuần mới `reports/[24 - 10] - Báo cáo.tex` |
| 2026-10-17 | **Demo + báo cáo dùng bản 512 (DEC-031):** `configs/demo_web.yaml` cho M1/M2/M5
trỏ vào `results/exp00{2,3,6}_*_512` (`max_length: 512`, `best_checkpoint.pt`); M3/M4/B1
giữ nguyên (không có biến thể 512). `scripts/score_custom_suite.py` thêm `--merge`,
`scripts/evidence_custom_suite.py` thêm `--reuse-moss` (máy thiếu `MOSS_USER_ID` thì giữ
artifact MOSS thật); `tests/test_demo_registry.py` chốt bản 512 bằng test. Chấm lại
`data/custom_suite` cho M1/M2/M5 rồi cập nhật §4 báo cáo `[17 - 10]` |
| 2026-10-17 | **Báo cáo tuần `[17 - 10]`:** đánh số mục lại liên tục (bỏ `\setcounter` hack cũ);
§5 *Kiến trúc suy luận phương pháp M2 (GraphCodeBERT)* đặt sau phần demo, dùng ảnh
`reports/images/Pipeline.png` (tạo từ prompt `reports/figures/m2_pipeline_prompt.md`, số liệu
trong ảnh khớp bản M2-512); §6 Hạn chế và §7 Kế hoạch tiếp theo để trống ruột, giữ tiêu đề.
Bản vẽ TikZ `pipeline_overview*` bỏ cùng lúc với mục *Pipeline chọn phương pháp* (nội dung
quy trình chọn nay nằm ở câu mở đầu §5) |
| 2026-10-17 | **Pipeline cuối (EXP-FINAL-001, DEC-030):** `configs/final_pipeline.yaml`,
`src/evaluation/leaderboard.py` (xếp hạng + chọn theo val F1, McNemar + paired bootstrap
tái dùng `uncertainty.py`), `scripts/final_pipeline.py` (`select`/`detect`),
`tests/test_final_pipeline.py`, artifact `results/final_pipeline/`,
report `reports/014_final_selection.md` |
| 2026-10-08 | M5 (Hybrid) vào demo: `src/demo/scorers/hybrid_scorer.py` (kế thừa CpgScorer,
tái dùng `extract_graphs`, evidence saliency nhánh text), `configs/demo_web.yaml`
thêm M5 (order + preload), `tests/test_demo_registry.py` lên 6 model |
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
