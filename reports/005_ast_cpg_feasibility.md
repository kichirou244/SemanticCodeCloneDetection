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

## Next Actions
1. Phase sau: AST extractor thật bằng tree-sitter → M3; CPG-Python → nửa M4.
2. User quyết định cài Joern khi tới M4-C++.
