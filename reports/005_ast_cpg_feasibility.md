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
1. Phase này: M3 (EXP-004) + M2 (EXP-003) trong một phiên Kaggle — DEC-023.
2. Phase sau: nếu muốn M4, cài JRE 17 + Joern, chạy `joern-parse` trên 500 file đo số
   giây/file rồi báo lại để quyết định ngân sách giờ.
