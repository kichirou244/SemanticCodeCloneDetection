"""Tạo notebook semanticcodeclonedetection.ipynb tự chứa (self-contained payload), giải quyết triệt để lỗi private GitHub token."""

import base64
import io
import json
from pathlib import Path
import tarfile


def build_repo_payload() -> str:
    """Nén thư mục src/ và scripts/ thành base64 tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("src", arcname="src", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("scripts", arcname="scripts", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_ast_cache_nb():
    payload_b64 = build_repo_payload()
    print(f"Kích thước mã nguồn đóng gói: {len(payload_b64) / 1024:.1f} KB")

    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title",
            "metadata": {},
            "source": [
                "# BIGCLONEBENCH — BƯỚC 1.1: XÂY DỰNG AST CACHE CHO M3 (ASTNN)\n",
                "\n",
                "Notebook này thực hiện nhiệm vụ tiền xử lý và trích xuất đặc trưng cây cú pháp (AST) cho toàn bộ tập dữ liệu **BigCloneBench Java** (`google/code_x_glue_cc_clone_detection_big_clone_bench`).\n",
                "\n",
                "### Cơ chế hoạt động & Tối ưu hóa:\n",
                "1. **Tự chứa (Self-contained):** Mã nguồn dự án mới nhất được nhúng sẵn dưới dạng bundle giải nén tức thì, không bị chặn bởi GitHub Authentication.\n",
                "2. **Deduplication (Gom hàm duy nhất):** Stream dữ liệu từ Hugging Face và gom thành ~25.000 – 50.000 hàm Java duy nhất theo `id1`/`id2`.\n",
                "3. **Tree-sitter Java Parser:** Parse toàn bộ mã nguồn Java ở tốc độ C native (~1.500 - 2.500 hàm/giây).\n",
                "4. **Biên dịch RvNN:** Biên dịch cấu trúc cây câu lệnh và lưu phẳng dạng `.npz` nén.\n",
                "5. **Đầu ra tại `/kaggle/working/`:**\n",
                "   - `bcb_ast_cache.npz`: File cache AST toàn bộ các hàm duy nhất (~30–50 MB).\n",
                "   - `bcb_ast_manifest.json`: Báo cáo chi tiết số lượng hàm, thời gian và checksum.\n",
                "\n",
                "Sau khi notebook chạy xong, bạn có thể thêm output này vào dataset `semanticcodeclonedetection-dataset-m4`."
            ],
        },
        {
            "cell_type": "code",
            "id": "c1-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ\n",
                "import base64\n",
                "import io\n",
                "import json\n",
                "import os\n",
                "import subprocess\n",
                "import sys\n",
                "import tarfile\n",
                "import time\n",
                "from pathlib import Path\n",
                "\n",
                "def sh(cmd, **kw):\n",
                "    print(f\"$ {cmd}\")\n",
                "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)\n",
                "    if r.stdout:\n",
                "        print(r.stdout.strip())\n",
                "    if r.stderr:\n",
                "        print(\"[stderr]\", r.stderr.strip())\n",
                "    if r.returncode != 0:\n",
                "        raise RuntimeError(f\"Lệnh thất bại (rc={r.returncode}): {cmd}\")\n",
                "    return r.stdout\n",
                "\n",
                "print(\"Môi trường Python sẵn sàng:\", sys.version.split()[0])"
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack-repo",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIẢI NÉN MÃ NGUỒN DỰ ÁN MỚI NHẤT (SELF-CONTAINED BUNDLE)\n",
                "WORK_DIR = Path(\"/kaggle/working/project\")\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                f"BUNDLE_B64 = {json.dumps(payload_b64)}\n",
                "\n",
                "print(f\"Đang giải nén mã nguồn bundle ({len(BUNDLE_B64) / 1024:.1f} KB) vào {WORK_DIR}...\")\n",
                "buf = io.BytesIO(base64.b64decode(BUNDLE_B64))\n",
                "with tarfile.open(fileobj=buf, mode=\"r:gz\") as tar:\n",
                "    tar.extractall(WORK_DIR)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                "\n",
                "print(\"Thư mục làm việc hiện tại:\", os.getcwd())\n",
                "sh(\"ls -la src/ representations/ast/ 2>/dev/null || ls -la src/\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-install-deps",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT CÁC THƯ VIỆN CẦN THIẾT\n",
                "sh(\"pip install -q 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn'\")\n",
                "\n",
                "# Kiểm tra Tree-sitter Java Parser\n",
                "from src.representations.ast.extract import get_parser\n",
                "p = get_parser(\"java\")\n",
                "sample_tree = p.parse(b\"public class Test { public static void main(String[] args) { int a = 10; } }\")\n",
                "print(\"tree-sitter-java sẵn sàng! Node count test:\", sample_tree.root_node.descendant_count)\n",
                "assert sample_tree.root_node.descendant_count > 10, \"Tree-sitter parser Java chưa hoạt động đúng!\""
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-stream-dedup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. STREAMING DỮ LIỆU TỪ HUGGING FACE & DEDUPLICATE HÀM JAVA DUY NHẤT\n",
                "from src.data.loaders.bigclonebench import BigCloneBenchLoader\n",
                "\n",
                "loader = BigCloneBenchLoader()\n",
                "unique_funcs = {}  # func_id -> source_code\n",
                "split_counts = {}\n",
                "t0 = time.perf_counter()\n",
                "\n",
                "splits_to_process = [\"train\", \"validation\", \"test\"]\n",
                "print(\"Bắt đầu stream dữ liệu BigCloneBench cho 3 split:\", splits_to_process)\n",
                "\n",
                "for s in splits_to_process:\n",
                "    s_t0 = time.perf_counter()\n",
                "    count = 0\n",
                "    print(f\"\\n--- Đang stream split '{s}' từ Hugging Face ---\")\n",
                "    for pair in loader.stream_pairs(split=s):\n",
                "        count += 1\n",
                "        sid1 = str(pair.id1)\n",
                "        sid2 = str(pair.id2)\n",
                "        if sid1 not in unique_funcs and pair.func1:\n",
                "            unique_funcs[sid1] = pair.func1\n",
                "        if sid2 not in unique_funcs and pair.func2:\n",
                "            unique_funcs[sid2] = pair.func2\n",
                "        \n",
                "        if count % 100000 == 0:\n",
                "            print(f\"  [{s}] Đã đọc {count:,} cặp | Hiện có {len(unique_funcs):,} hàm duy nhất ({time.perf_counter() - s_t0:.1f}s)\")\n",
                "    \n",
                "    split_counts[s] = count\n",
                "    print(f\"  [{s}] Xong split '{s}': {count:,} cặp (mất {time.perf_counter() - s_t0:.1f}s)\")\n",
                "\n",
                "total_pairs = sum(split_counts.values())\n",
                "total_unique = len(unique_funcs)\n",
                "el = time.perf_counter() - t0\n",
                "ratio = round((total_pairs * 2) / max(total_unique, 1), 2)\n",
                "\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"KẾT QUẢ THU THẬP & DEDUPLICATION:\")\n",
                "print(f\"  Tổng số cặp xử lý     : {total_pairs:,} {split_counts}\")\n",
                "print(f\"  Số hàm Java duy nhất  : {total_unique:,}\")\n",
                "print(f\"  Tỉ lệ nén / tiết kiệm : {ratio}x\")\n",
                "print(f\"  Thời gian stream      : {el:.1f} giây\")\n",
                "print(\"=\" * 60)"
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-build-ast-cache",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. XÂY DỰNG AST CACHE BẰNG TREE-SITTER VÀ LƯU VÀO /kaggle/working/\n",
                "from src.representations.ast.store import ASTTreeStore\n",
                "\n",
                "OUTPUT_CACHE_PATH = Path(\"/kaggle/working/bcb_ast_cache.npz\")\n",
                "print(\"Bắt đầu biên dịch cây cú pháp AST cho M3 (ASTNN)...\")\n",
                "t0 = time.perf_counter()\n",
                "\n",
                "# build() sẽ parse, compile statement trees và lưu trực tiếp ra file .npz\n",
                "store = ASTTreeStore.build(\n",
                "    codes=unique_funcs,\n",
                "    language=\"java\",\n",
                "    cache_path=OUTPUT_CACHE_PATH,\n",
                "    verbose=True,\n",
                ")\n",
                "compile_s = time.perf_counter() - t0\n",
                "\n",
                "file_size_mb = OUTPUT_CACHE_PATH.stat().st_size / (1024 * 1024)\n",
                "print(f\"\\nBiên dịch hoàn tất: {len(store):,} hàm trong {compile_s:.1f}s ({len(store)/compile_s:,.0f} hàm/s)\")\n",
                "print(f\"File cache đã lưu: {OUTPUT_CACHE_PATH} ({file_size_mb:.2f} MiB)\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-verify-cache",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. KIỂM TRA CHẤT LƯỢNG & TÍNH TOÀN VẸN CỦA AST CACHE\n",
                "import numpy as np\n",
                "import hashlib\n",
                "\n",
                "print(\"Đang nạp lại cache từ đĩa để xác thực...\")\n",
                "loaded_store = ASTTreeStore.build({}, cache_path=OUTPUT_CACHE_PATH, verbose=True)\n",
                "assert len(loaded_store) == len(unique_funcs), f\"Sai lệch số lượng: {len(loaded_store)} != {len(unique_funcs)}\"\n",
                "\n",
                "# Lấy thử 5 mẫu ngẫu nhiên kiểm tra cấu trúc mảng\n",
                "sample_ids = list(unique_funcs.keys())[:5]\n",
                "batch_compiled = [loaded_store[sid] for sid in sample_ids]\n",
                "assert len(batch_compiled) == 5\n",
                "\n",
                "for i, item in enumerate(batch_compiled):\n",
                "    assert \"top\" in item and \"s_label\" in item and \"anc_s\" in item\n",
                "    assert item[\"top\"].shape[0] >= 1, f\"Mẫu {sample_ids[i]} không có top node!\"\n",
                "\n",
                "# Test collate theo lô tương thích với mô hình M3 ASTNN\n",
                "from src.representations.ast.extract import collate_trees\n",
                "batch_np = collate_trees(batch_compiled)\n",
                "print(\"Test collation 5 mẫu thành công!\")\n",
                "print(\"  s_label shape :\", batch_np[\"s_label\"].shape)\n",
                "print(\"  s_depth shape :\", batch_np[\"s_depth\"].shape)\n",
                "print(\"  top_idx shape :\", batch_np[\"top_idx\"].shape)\n",
                "print(\"  top_mask shape:\", batch_np[\"top_mask\"].shape)\n",
                "\n",
                "# Tính sha256 checksum của file cache\n",
                "sha256 = hashlib.sha256(OUTPUT_CACHE_PATH.read_bytes()).hexdigest()\n",
                "print(f\"SHA256 checksum: {sha256}\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-manifest-summary",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. TẠO MANIFEST & TỔNG KẾT FILE ĐẦU RA\n",
                "manifest = {\n",
                "    \"task\": \"BCB-03 Build AST Cache for M3 ASTNN\",\n",
                "    \"created_at\": time.strftime(\"%Y-%m-%d %H:%M:%S\"),\n",
                "    \"dataset\": \"google/code_x_glue_cc_clone_detection_big_clone_bench\",\n",
                "    \"language\": \"java\",\n",
                "    \"splits\": split_counts,\n",
                "    \"total_pairs\": total_pairs,\n",
                "    \"unique_functions\": total_unique,\n",
                "    \"reduction_ratio\": ratio,\n",
                "    \"cache_file\": \"bcb_ast_cache.npz\",\n",
                "    \"cache_size_bytes\": OUTPUT_CACHE_PATH.stat().st_size,\n",
                "    \"sha256\": sha256,\n",
                "    \"status\": \"SUCCESS\",\n",
                "}\n",
                "\n",
                "manifest_path = Path(\"/kaggle/working/bcb_ast_manifest.json\")\n",
                "manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding=\"utf-8\")\n",
                "\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"TỔNG KẾT TÀI NGUYÊN TẠI /kaggle/working/:\")\n",
                "print(\"=\" * 60)\n",
                "sh(\"ls -lh /kaggle/working/bcb_ast*\")\n",
                "print(\"\\nManifest:\\n\", json.dumps(manifest, indent=2))\n",
                "print(\"\\nBƯỚC 1.1 HOÀN TẤT XUẤT SẮC! CÁC FILE ĐÃ SẴN SÀNG ĐỂ THÊM VÀO DATASET!\")"
            ],
        },
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("kaggle_tmp/semanticcodeclonedetection.ipynb")
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi notebook {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    create_ast_cache_nb()
