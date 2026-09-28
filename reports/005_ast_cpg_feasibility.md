# 005 — AST/CPG feasibility (GRAMMARS OK — còn Joern)

## Checklist
- `python scripts/check_toolchain.py` → tree-sitter + grammars cpp/python.
- Joern CLI cho CPG C/C++: `joern --version` (người dùng chạy tay).
- Python AST stdlib: luôn khả dụng (baseline cho M3-Python).

## Kết quả (2026-09-20)
- tree-sitter core: **có**.
- grammar cpp: **có** (đã cài).
- grammar python: **có** (đã cài).
- joern: **chưa cài** (`not recognized` trên PowerShell 2026-09-20).
  → Thứ tự implement theo DEC-009: AST cả 2 NN → CPG-Python → CPG-C++ sau.
- Transformer smoke (liên quan M1/M2/M5): CodeBERT embedding (2, 768) OK;
  GraphCodeBERT (2, 768) OK (cảnh báo `lm_head UNEXPECTED / pooler MISSING`
  là bình thường khi nạp `AutoModel` từ checkpoint pretrain — không ảnh hưởng
  embedding, sẽ đóng băng/phân loại ở phase sau).
- Baseline smoke B1: scores `[0.544, 0.182, 0.639]`, F1 = 1.0 trên 3 cặp mẫu
  (tín hiệu định tính tốt: cặp cùng ngữ nghĩa > cặp khác ngữ nghĩa).

## Options nếu CPG chưa khả thi
1. Hoãn M4/M5, chạy trước B1/M1/M2/M3-Python.
2. CPG-Python trước (ast+CFG tự xây), CPG-C++ sau khi có Joern.
3. Không thay M4 bằng model khác khi chưa có DEC và yêu cầu rõ ràng.

## Blockers
- Joern chưa cài (cần Java); M4-C++ chờ. Không chặn M3 và CPG-Python.

## Cập nhật 2026-09-28 — EXP-005 (M4 CPG-GNN) chưa làm được, và đo được vì sao

Kiểm lại trên máy user (không phải suy đoán từ tài liệu):

| Kiểm tra | Kết quả |
| :--- | :--- |
| `joern --version` | **không có** (vẫn `not recognized`, đúng như 2026-09-20) |
| `java -version` | **không có JRE** ⇒ không cài được Joern ngay |
| `python scripts/check_toolchain.py` | tree-sitter + grammar cpp OK; `cpg.status = feasibility_pending` |
| Code hiện có cho CPG | `src/representations/cpg/build.py` **chỉ là stub** `cpg_status()` — chưa có extractor, chưa có model, chưa có config |

**Throughput tree-sitter trên POJ-104 (đo, không ước lượng):** 500 file C++ thật →
**0,16 s** ⇒ **3.153 file/s**, trung bình 825 node/file. Toàn bộ POJ-104 (~62k code ở ba
split) parse hết trong **~20 giây**. Đây là lý do M3 (ASTNN) rẻ.

**Ước lượng Joern:** Joern cần JVM, front-end C/C++ đầy đủ (AST + CFG + data-flow) và
mỗi lần launch đều tốn thời gian khởi tạo. Thực tế Joern xử lý khoảng **0,3–1 file/s**
mỗi tiến trình. Với ~62k code:

- 1 tiến trình: **5,4–17,2 giờ**
- 4 tiến trình (Kaggle có ~4 vCPU): **1,4–4,3 giờ**

Tức là CPG **ăn hết hoặc gần hết** một phiên Kaggle 12h, chưa kể thời gian implement mô
hình CPG-GNN (node/edge embedding, GGNN/DRNN, ghép với cặp pair) chưa tồn tại trong repo.

**Kết luận:** EXP-005 **không** gộp được vào cùng phiên Kaggle với EXP-003/EXP-004
(DEC-023). Đây là milestone riêng: cài JRE + Joern → đo throughput thật trên 500 file →
mới quyết định có làm M4 hay không. Không đổi M4 sang model khác (giữ nguyên DEC-009).

**Next actions (cập nhật)**
1. Phase này: M3 (EXP-004) + M2 (EXP-003), mỗi mô hình một notebook — DEC-023.
2. Phase sau: M4 theo DEC-024 — **không cài Joern ở local**, cài trong
   `notebooks/exp005_cpg_extract_kaggle.ipynb`, đo 20 rồi 500 hàm, rồi mới quyết
   định ngân sách giờ.

## Cập nhật 2026-09-28 (lần 2) — M4 đã làm được phần không cần Joern (DEC-024)

Câu hỏi của người dùng: *"train ở Kaggle thì có cần test/cài ở local không?"*
**Không.** Lý do kỹ thuật: phần bị chặn không phải code M4 mà là **một lệnh cài**;
phần còn lại không cần Java thì viết và test được ngay.

| Tầng | File | Trạng thái | Test |
| :--- | :--- | :--- | :--- |
| Vocabulary (hợp đồng) | `src/representations/cpg/schema.py` | 13 nhãn node, 9 nhãn cạnh, 1 feature (`log1p(#args)/5`) | có |
| Nạp CPG | `src/representations/cpg/ingest.py` | JSON → `CPGraph`; cắt theo BFS từ `METHOD`; đếm mọi thứ bị bỏ | có |
| Đồ thị | `src/representations/cpg/graph.py` | `CPGraph` + `validate()` + `collate` PyG + `graph_stats` | có |
| Kho/cache | `src/representations/cpg/store.py` | `CPGStore` (`.npz` phẳng + con trỏ, giống `ASTTreeStore`) | roundtrip **phải bằng đúng** graph gốc |
| Model | `src/models/cpg_gnn/model.py` | GAT 3 tầp × 4 head, cạnh có nhãn, readout `[mean;max]`, head giống M1/M2/M3 | 204.801 tham số; bất biến hoán vị; đổi nhãn cạnh ⇒ đổi vector |
| Điều phối | `scripts/build_cpg_export.py` | lô + `--resume` + `benchmark.json` | test với **Joern giả** (không cần Java) |
| Exporter | `scripts/joern/export_cpg.sc` | chạy bởi Joern (JVM, Scala) | ⚠ **chưa kiểm chứng** — cần Kaggle |
| Smoke | `scripts/joern/joern_smoke.sc` | 1 hàm, in `cpg.help` + probe từng step | ⚠ **chạy trước exporter, bắt buộc** |
| Config | `configs/exp005_cpg_gnn.yaml` | `status: BLOCKED_ON_JOERN`, `blocked_by: [java, joern]` | có |

**33 test xanh local** (`tests/test_cpg_gnn.py`, `tests/test_cpg_scripts.py`), chạy được
trên máy không có Java. Hai lỗi thật đã bị test bắt trong lúc làm và đã sửa — ghi lại
vì cả hai đều là loại lỗi "chạy vẫn ra số":

1. `Data(x=numpy.ndarray)` ⇒ `Batch.from_data_list` gom thành `list` ⇒ `nn.Linear` ném
   lỗi `input must be Tensor, not list` (khóa bằng `test_to_pyg_fields_are_tensors`).
2. `edge_index` nhét vào `Data` phải là `(2, E)`; đưa nhầm `(E, 2)` thì collate cộng
   sai chiều và chết ngay khi ghép batch đầu tiên.

### Cài ở đâu (không cài local)

| Nơi | Có Java? | Quyết định |
| :--- | :--- | :--- |
| Máy dự án | **không** (`java -version` → không có) | **không cài** — cài tốn hàng trăm MB, đụng môi trường, và lần train vẫn ở Kaggle |
| Kaggle notebook | có sẵn JDK 17 + quota CPU | **cài ở đây**: `apt-get install openjdk-17-jre-headless` + `joern-install.sh` (chính thức, tự dò platform, đặt binary ở `~/bin/joern`) |

> Joern hiện là **v4.0.x** (bản mới hằng ngày), asset `joern-cli-linux-x86_64.zip`
> ~2GB. Cố tình **không** ghim version/tên file trong repo: bản cũ hơn vài tuần là
> 404, và người đọc không biết vì sao. Script cài chính thức tự dò đúng bản.

### Sửa thiết kế sau khi đối chiếu tài liệu (DEC-025)

Ba giả định ban đầu ở trên **đều sai**, sửa như sau:

| Giả định cũ | Thực tế | Cách sửa |
| :--- | :--- | :--- |
| `export_cpg.py` chạy bằng Python + `joernpy` | v4 chạy script bằng **Scala**: `joern --script x.sc --param k=v` | viết lại thành `export_cpg.sc`; xoá file `.py` |
| tải `joern.zip` của version ghim sẵn | tên asset theo version, bản mới hằng ngày | dùng `joern-install.sh` |
| một tiến trình Joern cho mỗi hàm | mỗi lần gọi `joern` = JVM + scala-repl, **vài chục giây** | gom `--batch-size` file vào một thư mục + `index.tsv`, gọi Joern **một lần cho cả lô** |

Hệ quả cho ngân sách thời gian: 62.000 hàm × 1 tiến trình × ~20s khởi động JVM
≈ **15 ngày** chỉ để bật JVM. Gom lô 100 hàm/lần ⇒ 620 lần khởi động cho cả
dataset. Vì vậy `--batch-size` là tham số quan trọng nhất của cell đo, không phải
một tiện ích.

`index.tsv` là bắt buộc: Joern chỉ trả về *tên file* (`00042.cpp`), còn
`sample_id` của dataset có dạng `poj104-train-7` — không thể suy ra.

### Quy trình đo (chạy trên Kaggle)

0. **Cell smoke (bắt buộc)** — `python scripts/build_cpg_export.py --smoke`:
   chạy `joern_smoke.sc` trên 1 hàm, in ra danh sách step CPGQL Joern tự khai báo
   + các dòng probe `ast=`, `callOut=`, `reachingDef=`, `methodReturn=`,
   `cfgNode=`, `parameter=`. **Nếu thiếu dòng nào thì dừng** — exporter sẽ không
   chạy được, và sửa sau khi đã bật cả 500 hàm là lãng phí.
1. `--limit 20 --batch-size 20` → xem `functions_per_second` và JSON đầu tiên có
   đúng là CPG không.
2. `--limit 500 --batch-size 100` → `benchmark.json`: `functions_per_second`,
   `error_rate` (gộp cả hàm không ra JSON), `nodes_per_cpg_mean`,
   `files_with_multiple_methods`, `est_hours_full_dataset`.
3. **Ngưỡng quyết định:** `< 0,3 hàm/s` ⇒ 62k hàm > 1 ngày ⇒ phải bàn lại phạm vi
   (giảm số hàm, hoặc song hoá 4 tiến trình) **trước khi** chạy full.
4. Gate chất lượng sau khi nạp cache: `nodes_per_code_mean < 20` ⇒ exporter hỏng;
   `files_with_multiple_methods > 0` ⇒ có thể đang lấy **sai hàm** (POJ-104 mỗi
   dòng là một hàm); `edges_dropped_*` lớn ⇒ lệch schema; thiếu nhãn
   `REACHING_DEF` ⇒ không còn là CPG.

**Ước lượng cũ (0,3–1 hàm/s ⇒ 5,4–17,2 h) vẫn giữ làm tham chiếu cho tới khi có số đo
thật** — khác biệt quan trọng: `build_cpg_export.py` đo trên **cùng một tiến trình
tuần tự**, nên con số đó là chi phí thật nếu chạy y hệt trong một phiên.

### Chưa làm (cố ý)

- Runner `src/experiments/exp005_cpg_gnn.py` + notebook train M4: chờ số đo, vì chúng
  chỉ khác EXP-004 ở chỗ load CPG store; viết trước rồi sửa batch size theo số node
  thật là làm hai lần.
- Chưa song hoá 4 tiến trình: giữ nguyên kiểu chạy thật để số đo đáng tin.
- Chưa chạy 62k hàm: 5–17 h là **quyết định ngân sách**, không phải việc làm mù.
