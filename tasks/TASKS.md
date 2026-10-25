# Task log — SemanticCodeCloneDetection

## [TO DO] Kế hoạch tuần 25-10 (CN) → 31-10 (T7): độ đo + non-clone + báo cáo tổng hợp
**Ghi chú từ yêu cầu ngày 25-10:**
- Nghiên cứu thêm các **độ đo trùng lặp** và giải thích rõ ràng (công thức, trực giác,
  ưu/nhược, khi nào dùng, đọc số thế nào).
- **Lấy thêm nhiều cặp không trùng lặp** — bộ 89 cặp hiện có 64 non-clone; cần dày hơn nữa
  cho sát thực tế và để F1 không bị chi phối bởi nhóm clone dễ.
- **Gộp báo cáo từ trước đến nay** khi báo cáo: một báo cáo tổng hợp duy nhất thay vì
  nhiều báo cáo tuần rời rạc.

**Mục tiêu tuần:** có tài liệu độ đo dùng được lâu dài; bộ non-clone ≥100 cặp và đã chấm
lại; một báo cáo tổng hợp 001–014 + các báo cáo tuần, số liệu khớp nguồn JSON.

### 1. Nghiên cứu độ đo trùng lặp + viết giải thích (25-10 → 26-10)
- Rà soát và phân loại độ đo theo 3 tầng:
  - **Mức cặp:** similarity score, ngưỡng, TP/FP/TN/FN, precision, recall, F1, accuracy,
    MCC, ROC-AUC/PR-AUC, calibration (Brier/ECE), bootstrap CI, McNemar.
  - **Giữa hai đoạn code:** token (Jaccard, TF-IDF cosine, winnowing/MOSS fingerprint,
    LCS/edit distance, NCD); AST (tree edit distance, subtree hash, ASTNN embedding);
    graph (graph edit distance, WL kernel, GNN embedding); embedding (cosine
    CodeBERT/GraphCodeBERT).
  - **Theo loại clone Type-1..4:** độ đo nào bắt được loại nào, loại nào là điểm mù.
- Giải thích rõ vì sao **điểm thô giữa B1/M1–M5 không so sánh trực tiếp được** (khác thang
  điểm, khác ngưỡng, khác mục tiêu huấn luyện) và cách báo cáo đúng: P/R/F1 tại ngưỡng
  đóng băng + khoảng bất định, tách "điểm trung bình ± SD của điểm thô" khỏi "F1 ± SD
  bootstrap".
- **Đầu ra:** `docs/clone_metrics.md` (công thức + trực giác + ưu/nhược + ví dụ lấy từ
  `custom_suite`), bảng map **độ đo → mô hình → thang điểm → ngưỡng**, và đưa bản rút gọn
  vào báo cáo tổng hợp.

### 2. Lấy thêm nhiều cặp không trùng lặp (27-10 → 28-10)
- Mở rộng non-clone **64 → mục tiêu 100–120 cặp** (tổng 125–145), ưu tiên ca khó:
  (a) cùng bài khác lõi; (b) khác bài cùng khung/biến đổi; (c) cùng skeleton nhưng khác
  cấu trúc dữ liệu/điều kiện biên; (d) mẫu hard-negative lấy từ **FP thật** của M1–M5
  trên validation/test (dùng làm ý tưởng thiết kế, không copy code nếu vướng license).
- Mở rộng `scripts/build_custom_suite_negatives.py` (luật tất định, chặn trùng/tự thân/
  thiếu file), cập nhật `tests/test_custom_suite_manifest.py` + `data/custom_suite/README.md`.
- Chấm lại M1–M5 theo lô (`--only-new --batch-pairs`), giữ B1 41 cặp gốc; cập nhật
  `workspace/custom_suite_summary.{json,md,tex}`.
- **Tiêu chí xong:** ≥100 non-clone, không cặp trùng, mọi file biên dịch sạch, nhãn được
  kiểm chứng, có bảng delta FP/F1 so với bản 89 cặp.

### 3. Gộp báo cáo từ trước đến nay (29-10 → 30-10)
- Tạo **một báo cáo tổng hợp** (dự kiến `reports/[31 - 10] - Báo cáo tổng hợp.tex` + PDF),
  gộp `001–014` và các báo cáo tuần `[03-10]`, `[06-10]`, `[09-10]`, `[10-10]`, `[17-10]`,
  `[24-10]`:
  1. Bài toán + phạm vi C/C++.
  2. Dữ liệu POJ-104: audit, leakage, dedup, pair generation.
  3. Sáu phương pháp B1/M1–M5 + kiến trúc, checkpoint, ngưỡng.
  4. Quá trình thực nghiệm 001–014 + bảng kết quả validation/test.
  5. Lịch sử bộ `custom_suite` 26 → 34 → 41 → 89 cặp và lý do từng lần đổi.
  6. Phân tích lỗi: FP/FN, truncation 512, "bám khung", khoảng cách điểm.
  7. Độ đo và cách đọc số (trỏ `docs/clone_metrics.md`).
  8. Hạn chế + kế hoạch tiếp theo.
- Số liệu lấy từ **một nguồn duy nhất** (`workspace/*.json`, `results/*/metrics.json`,
  `reports/014_final_selection.md`), không chép lại prose cũ. Kèm **phụ lục truy vết**:
  mỗi báo cáo cũ → mục tương ứng. **Không sửa/xoá báo cáo cũ.**

### 4. Chốt số + build PDF + cập nhật tài liệu (31-10, T7 nộp)
- Chốt bộ test dùng cho báo cáo (89 hoặc bản mở rộng), ghi rõ từng bảng đang tính trên
  bao nhiêu cặp.
- Build 2 lượt xelatex (**0 error / 0 Overfull**), kiểm tra bảng/biểu đồ vừa trang.
- Cập nhật `data/custom_suite/README.md`, `docs/decision_log.md` (DEC mới cho mở rộng
  non-clone + metric glossary), `project_structure.md`, `tasks/TASKS.md`.
- Kiểm tra chéo: mọi số trong báo cáo tổng hợp khớp `workspace/custom_suite_summary.json`
  và `reports/014_final_selection.md`.

## [COMPLETE] Mở rộng non-clone 16 → 64 cặp, tổng 89 cặp (25-10) — DEC-034
- Bộ 41 cặp chỉ có 16 non-clone, không đại diện đầu vào thực tế → thêm **48 cặp non-clone**
  sinh tất định bằng `scripts/build_custom_suite_negatives.py`: 24 cùng bài khác lõi,
  6 base–base còn thiếu, 18 khác bài cùng biến đổi. Tổng **89 cặp: 25 clone + 64 non-clone
  (4,0× bản cũ)**, giữ nguyên 41 cặp và id cũ.
- Chấm theo lô (`--only-new --batch-pairs 16`; Joern một lần cho M4/M5); B1 vẫn chỉ có
  41 cặp gốc nên so sánh công bằng trên 41 cặp chung.
- Kết quả 89 cặp: recall 1,000 cho cả 5 mô hình; F1 M2 0,694, M1 0,685, M5 0,658,
  M3 0,500, M4 0,446; dương giả non-clone 23/22/26/50/62 trên 64. Trên 41 cặp chung:
  M2 0,909, M1 0,893, M5 0,877, M3 0,794, M4 0,758, B1 0,400.
- Tổng hợp mean ± SD + F1 bootstrap: `scripts/summarize_custom_suite.py` →
  `workspace/custom_suite_summary.{json,md,tex}`; báo cáo `reports/[24 - 10] - Báo cáo.tex`
  (9 trang); bất biến manifest khoá bằng `tests/test_custom_suite_manifest.py`.

## [COMPLETE] Bộ test mở rộng nhóm P5: DSLK + hướng đối tượng (17-10) — DEC-032
- Thầy góp ý bộ test còn ngắn → thêm 6 file C++ **dài 59–70 dòng** vào `data/custom_suite/`:
  bản gốc `struct Node` + 4 hàm tự do (`p5_base_list`), **cùng hành vi** viết bằng
  `class IntList` + 4 phương thức (`p5_v_class`), 3 biến thể clone
  (`p5_v_while/scanf/combo`), và `p5_n_palindrome` (cùng khung `class`/`Node`, khác ngữ nghĩa).
- 8 cặp mới trong `pairs.jsonl` (26 → 34 cặp): `P5_struct_class` (struct ↔ class),
  `P5_while`, `P5_scanf`, `P5_combo`, `P5_class_combo`, `P5_nonclone_palindrome`,
  `X_P1P5`, `X_P2P5`.
- Kiểm chứng code trước khi chấm: 6/6 file biên dịch sạch (`g++ -std=c++17 -O2 -Wall`),
  5 bản tính tổng cho **cùng kết quả** trên cùng input, palindrome trả `NO`/`YES` đúng.
- Kết quả: **cả 5 model bắt đúng cặp struct↔class** (M1/M2/M5 ≈ 1,000; M3 0,915; M4 0,973),
  còn **MOSS bỏ sót cả hai** cặp clone dạng `class` (0,000). M3 **đúng cả 8 cặp**; M4 sai 3
  (toàn dương giả: `P5_nonclone_palindrome` 0,743 và hai cặp cross).
  Lỗi mới: `X_P1P5` (DSLK vs tổng 1..n) bị M1/M2/M5 ≈ 1,000 vì hai bài cùng phép toán tổng
  và cùng khung `main`.
- Phát hiện phụ: **cả 6 file P5 vượt 512 subword token** (520–683) nên M1/M2/M5
  (`max_length: 512`) chỉ đọc phần đầu, cắt tới **25 %** ở `p5_v_class`; M3/M4 không bị cắt
  (CPG 61–95 node so với trần 1500) → muốn test code dài hơn nữa thì phải nới `max_length`
  hoặc cắt theo hàm (đã hoãn từ DEC-029).
- Công cụ: `score_custom_suite.py --pairs`, `evidence_custom_suite.py --models` + `fig_font`
  (`\scriptsize` cho file dài), `_latex_minipage(font=)`, `.gitignore` mở ngoại lệ
  `!data/custom_suite/` để code test vào git.
- Báo cáo tuần mới `reports/[24 - 10] - Báo cáo.tex` (+ PDF, 6 trang), build 2 lượt xelatex,
  0 Overfull: 6 file + 8 cặp, trích code hai cột, bảng điểm 8 × 6, 2 hình minh hoạ M2.

## [COMPLETE] Dừng train → pipeline cuối: đánh giá + chọn + suy luận (17-10)
- Quyết định **dừng huấn luyện** (các bản 512 đã chạm trần thực dụng, chênh còn lại
  nằm trong nhiễu) và chuyển sang đánh giá để chốt phương pháp → DEC-030.
- Mới: `configs/final_pipeline.yaml`, `src/evaluation/leaderboard.py`,
  `scripts/final_pipeline.py` (`select`/`detect`), `tests/test_final_pipeline.py`
  (14 test, không cần GPU), `reports/014_final_selection.md`, artifact
  `results/final_pipeline/`.
- `select` xếp hạng 10 run theo **val F1** (ngưỡng đóng băng), so winner vs á quân
  bằng **McNemar + paired bootstrap tái dùng `uncertainty.py`** trên validation.
- Kết quả: **M5-512 thắng theo val F1** (0,9379) nhưng hơn M2-512 chỉ +0,0043;
  McNemar p=0,0814 (không ý nghĩa) vs bootstrap CI95 [+0,0004,+0,0083] (chạm 0) ⇒
  `mixed`; ngưỡng 0,010 rất mỏng; **M2-512 hơn ở test F1 0,9550 / AUC 0,9912** ⇒
  **pipeline cuối dùng M2-512 cho `detect`** (`inference.default_model: M2_512`;
  M5-512 gọi khi cần bằng `--model M5_512`).
- Kế hoạch train tuần 18–23/10 (Hybrid V3, CPG nới node, Hybrid-512) **tạm gác** —
  chỉ mở lại nếu cần bằng chứng mạnh hơn cho lựa chọn.
- Chạy lại: `python scripts/final_pipeline.py select` ·
  `python scripts/final_pipeline.py detect --a a.cpp --b b.cpp`

## [COMPLETE] Demo + báo cáo chuyển sang bản 512 cho M1/M2/M5 (17-10) — DEC-031
- `configs/demo_web.yaml`: M1/M2/M5 trỏ `results/exp00{2,3,6}_*_512` (`max_length: 512`,
  `best_checkpoint.pt`; ngưỡng vẫn đọc từ `threshold.json` của chính run). M3/M4/B1 giữ
  nguyên vì không có biến thể 512 (AST / node CPG / winnowing đều không phụ thuộc
  `max_length`).
- Chấm lại bộ test: `python scripts/score_custom_suite.py --models M1 M2 M5 --merge`
  (giữ B1/M3/M4) rồi `python scripts/evidence_custom_suite.py --reuse-moss --pairs
  P1_combo P3_nonclone_sum P1_nonclone_fact` (giữ lại artifact MOSS service thật).
- Kết quả trên 26 cặp: lỗi M2 3→**2**, M5 4→**3** (sửa 2 dương giả ở `P2_nonclone_digitsum`),
  **không cặp nào đổi từ đúng sang sai**; §4 báo cáo `[17 - 10]` cập nhật số mới.
- Guard: `tests/test_demo_registry.py::test_transformer_models_use_512_runs`.

## [QUY UOC] Bo test mac dinh: `data/custom_suite` (tu 16-10)
- Moi lan test/demo/bao cao sau nay deu dung `data/custom_suite`
  (89 cap: 25 clone + 64 non-clone, tu 25-10), cham bang `scripts/score_custom_suite.py`,
  sinh hinh bang `scripts/evidence_custom_suite.py`. Khong them comment vao file
  test (da xoa sach 16-10). `data/custom_hard/` giu lai lam tham khao.
- Ket qua full 16-10 (khong comment): 16/16 clone ca 5 model neural dung;
  non-clone kho: P1_fact + P4_square ca 5 model sai, P2_digitsum chi M1 dung,
  P3_sum M1/M2/M5 dung; cross X: M1/M2 6/6, M5 4/6, M3/M4 0/6. B1-service 0 diem
  moi cap (mu Type-3/4, da biet tu EXP-014).
- Bao cao [17-10]: muc 6 co 3 cap (6.1 clone kho P1_combo, 6.2 non-clone kho
  P3_nonclone_sum, 6.3 bay skeleton P1_nonclone_fact ca 5 model sai).
  Khong ton tai ca false-non-clone nhat tri (16/16 clone deu bi bat).

## [QUY UOC] Cập nhật trạng thái task trên Notion (từ 18-10)
- Trước khi bắt đầu 1 việc: mở task trong Notion, đổi Status -> **In progress**.
- Làm xong: đổi Status -> **Done** (ghi kết quả vào Description nếu cần).
- Xem task: `node tasks/notion.js --tasks`
- Đổi status: `node tasks/notion.js --status "<tên task>" "<trạng thái>"`
  (trạng thái: Not started / In progress / On Hold / To Review / Done / Cancelled)
- Sync kế hoạch từ `tasks/tasks.json` lên Notion: `node tasks/notion.js`
  (khớp theo tên: có thì cập nhật, chưa có thì tạo mới, không tạo trùng).

## [ARCHIVED] Kế hoạch tuần 18-10 (CN) → 23-10 (T6), T7 nộp báo cáo
> Đã đóng: nhóm P6 ≥120 dòng được thay bằng bản 83–93 dòng và chấm lại toàn bộ 41 cặp
> theo **DEC-033 (24-10)**. Giữ mục này để tra cứu lịch sử, không còn là việc đang làm.
**Đổi hướng theo feedback sau báo cáo `[17 - 10]`:** bộ test còn khá ngắn → tuần này
chọn thêm code **dùng hàm**, **cấu trúc DSLK**, **hướng đối tượng**; code dài và có
phần **giống nhau gần như hoàn toàn giữa `struct` và `class`** → làm **nhóm P6**.
Toàn bộ kế hoạch huấn luyện tuần trước (Hybrid V3, CPG nới node, Hybrid-512 cấu hình
mới, đo lại truncation) đã **chuyển sang `Cancelled` trên Notion** (khớp tinh thần
DEC-030: tạm dừng huấn luyện) — chỉ giữ lại phần truncation vì code dài >512 token
thì bắt buộc phải xử lý.

### 1. Nhóm test P6: 6 file ≥120 dòng (hàm + DSLK + OOP) — 18-10
- Soạn `data/custom_suite/p6_*`: base = hàm tách nhỏ (`nhap/xuLy/in`) + `struct Node`
  + DSLK; bản clone **cùng hành vi** viết bằng `class` + phương thức → cặp
  `P6_struct_class` đo đúng ca "struct ↔ class gần như giống hoàn toàn".
- Thêm biến thể `v_while` / `v_scanf` / `v_combo`, 1 non-clone khó cùng khung
  class/Node khác ngữ nghĩa, và cặp cross P6↔P5.
- Thêm 6–8 cặp vào `data/custom_suite/pairs.jsonl` (34 → ~41 cặp). Giữ quy ước:
  không comment trong file test.

### 2. Kiểm chứng + truncation 512 token — 19-10
- 6/6 file biên dịch sạch (`g++ -std=c++17 -O2 -Wall`), các bản clone cùng input →
  cùng kết quả; non-clone khác kết quả đúng thiết kế.
- Đo subword token/`truncation.json`: P5 đã 520–683 token nên P6 chắc chắn vượt
  `max_length: 512` → chọn nới `max_length` hoặc cắt theo hàm (hoãn từ DEC-029),
  ghi tỉ lệ bị cắt để báo cáo giải thích được.

### 3. Chấm điểm + evidence + phân tích lỗi — 20-10 → 21-10
- `python scripts/score_custom_suite.py --pairs <cặp P6> --models B1 M1 M2 M3 M4 M5 --merge`.
- `python scripts/evidence_custom_suite.py --pairs P6_struct_class P6_nonclone_*`.
- Phân tích: model nào bắt/miss cặp struct↔class dài, lỗi mới ở non-clone khó + cross,
  so với P5 (MOSS bỏ sót hai cặp `class`; M1/M2/M5 ≈ 1,000 nhưng bị cắt token).

### 4. Chốt số + báo cáo tuần (T7 nộp) — 22-10 → 23-10
- Cập nhật `reports/[24 - 10] - Báo cáo.tex`: mục nhóm P6 + bảng điểm 8 × 6 + hình,
  điền mục Hạn chế và Kế hoạch tiếp theo.
- Build PDF 2 lượt xelatex (0 error/0 Overfull), cập nhật `data/custom_suite/README.md`.

## [QUYET DINH] Huỷ 4 task huấn luyện trên Notion, dồn tuần cho bộ test dài (18-10)
- `Cancelled` trên Notion: Hybrid V3 (freeze→unfreeze), CPG-GNN nới node,
  Hybrid-512 cấu hình mới, Đo lại truncation 512.
- Task cũ "Chốt số & báo cáo tuần (T7 nộp)" được **cập nhật tại chỗ** (khớp theo tên,
  không tạo trùng), mô tả mới gắn với nhóm P6.
- 3 task P6 mới đã tạo trên Notion (Due 18/19/20-10), gắn relation project
  `[HUSC] NLP: SemanticCodeCloneDetection`.

## [COMPLETE] Bổ sung MOSS + ví dụ khó theo review báo cáo [10-10]
- Đào từ `test_scores.jsonl` (ngưỡng threshold.json): 11 cặp FN cả 5 mô hình,
  65 cặp FP cả 5 mô hình. Chọn `test-pos-30-284` (GT=clone, 0,073/0,022/0,407/
  0,292/0,002) và `test-neg-218` (GT=non-clone, 0,998/0,999/0,734/0,828/0,999).
- Tự tạo `data/custom_hard/`: `base_sum7.cpp` + 3 biến thể clone (for→while,
  cin→scanf, combo rename+do-while+dead code) + 1 non-clone khó (cùng skeleton,
  `cnt+=1` thay `sum+=i*i`). Chấm local M1/M2/M3 + MOSS-local (k=20,w=5):
  3 cặp clone đều đúng; cặp non-clone khó cả 4 đều sai (1,0/1,0/0,978/0,795).
- MOSS-local: FN POJ 0,186 (đúng — thắng cả 5 model), FP POJ 0,033 (sai cùng).
  M4/M5 cặp tự tạo chưa chấm local được (thiếu Java/Joern).
- Báo cáo: thêm hàng MOSS vào bảng test + mục 6.3–6.6, build PDF 8 trang
  (`[09-10]` .tex/.pdf, copy sang `[10-10]` .pdf).

## [COMPLETE] Review lần 2: gọn 6.3+, MOSS highlight, bộ test tái dùng
- Bỏ hàng MOSS khỏi bảng test (theo review). Minh họa MOSS = highlight k-gram
  chung render bằng pipeline evidence có sẵn (`svc.evidence_for("B1")` local +
  `_latex_minipage` trong `scripts/scan_demo_zip.py`) → `workspace/demo_moss_custom.tex`
  (2 cặp C6/N1), không gọi service Stanford.
- Bộ test mở rộng 8 cặp: `data/custom_hard/pairs.jsonl` (manifest) + 4 file mới
  (dowhile, reorder_dead, while_scanf, max_track) + `scripts/score_custom_hard.py`
  (chạy lại bất cứ lúc nào) → `workspace/custom_hard_scores.json`.
  Kết quả: 6 clone đúng hết (M3 0,777–0,963), 2 non-clone khó sai hết.
- Báo cáo viết lại 6.3 (bảng 8 cặp + cách tái dùng), 6.4 (bảng 2 ca POJ cả-5-sai),
  6.5 (MOSS highlight). Build PDF 8 trang, copy sang `[10-10]` .pdf.

## [COMPLETE] MOSS server thật cho bộ test khó (user cấp id)
- Sửa bug do đặt id: `_ENV_USER_ID` phải là tên biến môi trường, id vào
  `_DEFAULT_USER_ID` (env `MOSS_USER_ID` được ưu tiên nếu có).
- Nộp gộp 13 file / 1 lần lên `moss.stanford.edu`
  (report `http://moss.stanford.edu/results/9/6309937295477`):
  10 cặp chỉ 3 bị gắn cờ — C2 (0,61 đúng), N1 (0,75 sai), N2 (0,55 sai);
  mọi biến thể đổi vòng lặp + cả 2 cặp POJ đều 0 điểm.
- Báo cáo chuyển cột MOSS sang điểm server + highlight từ trang match Stanford
  (`workspace/demo_moss_custom.tex` viết lại, `moss_service_scores.json`).
  Kết luận: MOSS thật mù Type-3/4 (5/6 clone tự tạo + FN POJ đều 0 điểm).
- Minh họa 6.1/6.2 thêm MOSS server (`workspace/demo_moss_61.tex`,
  `demo_moss_62.tex` từ trang match Stanford: 6.1 → 0,980 đúng, 6.2 → 0,000 đúng).
  Sửa lỗi build LaTeX có từ trước (babel `"` trong code mẫu, `_` trong tiêu đề):
  build còn 0 error. PDF 10 trang, copy sang `[10-10]` .pdf.
- Theo yêu cầu: xóa mục 6.3–6.5 (bộ test tự tạo + 2 ca POJ + MOSS highlight custom).
  Báo cáo còn 6.1/6.2 (5 model + MOSS server), 8 trang, build 0 error.
  Bộ test `data/custom_hard/` + `scripts/score_custom_hard.py` vẫn giữ trong repo
  để dùng lại sau này.

## [COMPLETE] Hậu kỳ highlight saliency (Top-K + stop-words + merge vệt)
- Đo trên saliency thật M1 (2 cặp × 2 phía): baseline (>mean) cov 23–30%,
  punct 4–11%, 8–15 vệt; Top-15% → cov 15%, punct ~0%, 6–11 vệt.
  Gom theo dòng tuy ít vệt (3–6) nhưng cov phình 46–55% (lôi cả dấu ngoặc) nên loại.
  SmoothGrad (N=8) gần như không khác baseline nên loại (đắt gấp 8 lần).
- Mới: `src/demo/saliency.py` (`select_salient_spans`, quantile 0.85),
  dùng chung trong `transformer_scorer.py` (M1/M2) + `hybrid_scorer.py` (M5),
  note evidence ghi rõ quy tắc lọc. M3/M4/B1 không đụng.
- Kiểm chứng: M1 evidence cặp C6 còn 8/9 vệt, 0 span chỉ-dấu-câu, tất định;
  `test_demo_scan` + `test_demo_server` xanh (sửa 1 test MOSS fallback cho khớp
  `_DEFAULT_USER_ID` mới). Không sinh lại hình báo cáo (M4/M5 cần Joern).
- Sinh lại hình 6.1/6.2 với logic saliency mới (máy có Joern tại
  `D:\Code\SDK\joern`, truyền qua env `JOERN_BIN` khi chạy):
  `workspace/demo_clone5.tex` + `demo_neg5.tex` viết lại, điểm số giữ nguyên
  khớp báo cáo (M4/M5 evidence chạy được, M1 clone còn 5/5 vệt).
  Build PDF 8 trang 0 error, copy sang `[10-10]` .pdf.
- Gọn + chính xác: thêm `filter_pairs_by_spans` (`saliency.py`) — M1/M2/M5 chỉ
  giữ cặp mutual-best chạm vùng salient (6.1: 30 cặp → M1 còn 2, M2 còn 4,
  M5 còn 3–4; cặp top-8 xếp theo saliency vẫn lẫn boilerplate `#include` nên loại).
  Sinh lại hình + build PDF 0 error; chú thích "Cách đọc hình" cập nhật theo.
- Theo yêu cầu: revert toàn bộ về cách ban đầu (mean + đủ pairs):
  `git checkout` transformer/hybrid scorer + project_structure, xóa `saliency.py`,
  chú thích báo cáo về cũ, sinh lại hình (M1 26/25, M2 30/30 vệt như cũ),
  build PDF 8 trang 0 error, test xanh. Giữ lại fix test MOSS fallback.

## [COMPLETE] Báo cáo `[09 - 10]`
- Tạo `reports/[09 - 10] - Báo cáo.tex` từ `[06-10]-Report.tex`, xoá đoạn nhận xét sau bảng huấn luyện.
- Gộp mục 1+2 thành mục 1 `Kết quả tuần này` (hoàn thành train 5 PP + demo trên tập test ở mục 6).
- Sửa chữ không dấu: `(khong clone)` → `(không clone)` trong `workspace/demo_neg5.tex` (+ file txt đi kèm), `model` → `mô hình` trong báo cáo.
- Build PDF 7 trang xong.

## [COMPLETE] Tối ưu train V2 (vừa quota Kaggle 10h)
- M5 Hybrid (`configs/exp006_hybrid.yaml` + `src/experiments/exp006_hybrid.py`):
  LR head 2e-5 → 1e-5, thêm `encoder_learning_rate: 5e-6` (LR phân biệt encoder/head),
  `epochs` 5 → 3. Lý do: E0 đỉnh 0,914 rồi train loss sập 0,054 → 0,003 (overfit).
- M4 CPG-GNN (`configs/exp005_cpg_gnn.yaml`): `patience` 2 → 4, giữ `epochs: 15`
  (bản báo cáo dừng ở epoch 5, val F1 còn tăng 0,780 → 0,819).
- Test: 33 passed (`test_exp006_hybrid` + `test_configs` + `test_cpg_gnn`).
- Notebook Kaggle:
  - `exp005_cpg_gnn_kaggle.ipynb` cell 9 = FULL TRAIN từ đầu, budget 3,5h.
  - `exp006_hybrid_kaggle.ipynb` cell 10 = train MỚI hoàn toàn (không resume), budget 7h.

## [COMPLETE] M4 V2 + báo cáo (10/10)
- Train full 15 epoch: val F1 0,819 → **0,842** (E14), test F1 0,779 → **0,797**,
  ngưỡng mới 0,470. Train/val loss giảm mượt, không overfit.
- Chạy lại demo M4 mới (local, Joern): cặp clone 0,962 → **0,967**,
  cặp neg 0,294 → **0,190**. Tái sinh section CPG-GNN trong
  `workspace/demo_clone5.tex` + `demo_neg5.tex`.
- Báo cáo cập nhật: mục 1 (15 epoch), bảng train (hàng CPG mới + ghi tốt nhất E14),
  bảng test (0,723/0,887/0,797/0,882/0,470), câu 6.1/6.2. PDF build lại.

## [COMPLETE] Hybrid V2 -> verdict: KHÔNG tốt hơn, giữ V1 (10/10)
- V2 (3 epoch, LR nhỏ): val F1 0,914 -> 0,903 (kém), test F1 0,9133 -> 0,9138
  (hòa, trong nhiễu), ngưỡng sập 0,370 -> 0,010 (mỏng, đáng ngờ).
- Overfit vẫn còn (val E2 rớt 0,882 trong khi train loss vẫn giảm) — giảm LR
  chưa đủ, hướng tiếp theo là freeze encoder (chỉ train head).
- Khôi phục: `exp006_hybrid_poj104` = V1 (test 0,9133/thr 0,37), V2 cất sang
  `exp006_hybrid_poj104_v2`. Báo cáo giữ nguyên số V1, không cần sửa.

## [COMPLETE] Chuẩn bị train 512 token M1/M2 (warm-start từ 256)
- `transformer_pair_runner.py`: thêm `_maybe_warm_start()` — nạp `model_state`
  (strict, cùng kiến trúc, pos-embeds vốn đã 512), optimizer/scheduler reset.
  Chỉ kích hoạt khi config có `model.warm_start`, run cũ không ảnh hưởng.
- 2 config mới: `exp002_codebert_512.yaml` (M1) + `exp003_graphcodebert_512.yaml`
  (M2): `max_length` 256 → 512, `batch_size` 16 → 8, accum 2 → 4 (giữ
  effective batch 32), `pair_batch_size` 64 → 32, epochs 3 + patience 2.
- Checkpoint 256 (`best_weights_only.pt` M1/M2, ~506MB mỗi file) đã push lên HF
  `Kichirou24/SemanticCodeCloneDetection` (`m1/m2_pair_256_weights_only.pt`, LFS).
- 2 notebook mới `exp002_codebert_512_kaggle.ipynb` / `exp003_graphcodebert_512_kaggle.ipynb`
  (copy từ bản 256 + cell kéo weights HF về đúng path `warm_start`).
- Test: 33 passed (`test_configs` + `test_exp002` + `test_exp003`).
- Diagnostic trên Kaggle T4: M1 + M2 đều PASS (CUDA, gradient finite,
  params updated, loss giảm).

## [IN PROGRESS] Train M1/M2 512 trên Kaggle (user, warm-start từ 256)
1. Chạy 2 notebook 512 (M1 trước cho rẻ, rồi M2).
2. Khi xong: tải `results/exp00*_..._512/` về, so truncation mới
   (`truncation.json`, kỳ vọng rate rớt từ ~60-78% về gần 0) + val/test F1 vs bản 256.
3. Nếu ok mới xét M5-512 (text 512 + GNN, tốn VRAM nhất nên để sau cùng).

## [READY] M5 Hybrid-512 đấu M2-512 (config + notebook xong 13-10, chờ M2-512 về)
- `configs/exp006_hybrid_512.yaml`: max_length 512, warm-start text từ
  `results/exp003_graphcodebert_poj104_512/best_checkpoint.pt` + graph M4 giữ nguyên,
  batch 12->6 / accum 2->4 (effective ~24), pair_batch 32->16, LR V2
  (head 1e-5 / encoder 5e-6), out `results/exp006_hybrid_poj104_512`.
- `notebooks/exp006_hybrid_512_kaggle.ipynb`: copy từ bản 256, trỏ config/out mới,
  HF file `m2_512_text_encoder.pt` (+ `m4_graph_encoder.pt` giữ nguyên).
- Test: `test_configs` + `test_exp006_hybrid` + `test_demo_registry` xanh (15 passed).
- Việc còn lại (user): (1) export encoder M2-512 local
  (`export_hybrid_warm_start --m2 <m2-512 ckpt> --m4 <m4 ckpt>`), đổi tên
  `m2_text_encoder.pt -> m2_512_text_encoder.pt`, upload HF; (2) audit bổ trợ lại
  trên `test_scores.jsonl` M2-512 trước khi đốt ~8h train; (3) chạy notebook 512.

## [TO DO] Chạy trên Kaggle (user)
1. Import 2 notebook, `git pull` ở cell 2.
2. Chạy CPG trước (~3h), rồi Hybrid (~7-8h).
3. Backup `results/exp006_hybrid_poj104/best_checkpoint.pt` cũ nếu cần giữ số báo cáo 09-10
   (Hybrid train mới ghi đè thư mục kết quả).

## [PLAN] Sau khi có số mới
- Cập nhật bảng val F1 + test trong `reports/[09 - 10] - Báo cáo.tex`, build lại PDF.
- So sánh Hybrid V2 vs V1 (hết overfit E0 chưa), CPG epoch mới có tăng tiếp không.

## [BACKLOG] Ý tưởng & tính năng mở rộng (không đẩy lên Notion)
- **AI vấn đáp nhanh kiểm tra mức độ hiểu code của sinh viên:**
  - *Ý tưởng:* Khi phát hiện nghi vấn trùng lặp hoặc cần thêm cơ sở đánh giá độ hiểu mã nguồn, AI tự động phân tích code và sinh bộ câu hỏi vấn đáp nhanh (dạng trắc nghiệm nhiều lựa chọn hoặc Đúng/Sai).
  - *Mục tiêu:* Đánh giá xem sinh viên có thực sự hiểu cấu trúc, luồng thực thi và logic của đoạn code đã nộp hay chỉ sao chép/nhờ công cụ viết hộ; bổ sung thêm một chiều cơ sở dữ liệu định lượng bên cạnh điểm tương đồng của mô hình.

  - nghiên cứu thêm các độ đo trùng lặp
  - lấy thêm nhiều cặp k trùng lặp