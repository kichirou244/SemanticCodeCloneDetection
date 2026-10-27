"""Tạo notebook semanticcodeclonedetection.ipynb cho huấn luyện M3 (ASTNN) trên GPU Kaggle."""

import base64
import io
import json
from pathlib import Path
import tarfile


def build_repo_payload() -> str:
    """Nén thư mục src/, scripts/ và configs/ thành base64 tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("src", arcname="src", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("scripts", arcname="scripts", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("configs", arcname="configs", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_m3_nb():
    payload_b64 = build_repo_payload()
    print(f"Kích thước mã nguồn đóng gói (src + scripts + configs): {len(payload_b64) / 1024:.1f} KB")

    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title",
            "metadata": {},
            "source": [
                "# EXP-BCB-004 — HUẤN LUYỆN M3 (ASTNN) TRÊN BIGCLONEBENCH JAVA (GPU)\n",
                "\n",
                "Notebook này thực hiện huấn luyện mô hình mạng nơ-ron cây **ASTNN (M3)** trên tập dữ liệu **BigCloneBench Java** sử dụng tăng tốc **Kaggle GPU (T4 / P100)**.\n",
                "\n",
                "### Điểm cốt lõi:\n",
                "1. **Tận dụng AST Cache có sẵn:** Mô hình nạp trực tiếp các cây câu lệnh đã compile từ dataset `semanticcodeclonedetection-dataset-m4` (`bcb_ast_cache.npz`), **hoàn toàn loại bỏ nghẽn CPU** khi đóng batch.\n",
                "2. **Kiến trúc RvNN + BiGRU:** Mô hình gọn nhẹ (~428,000 tham số), huấn luyện cực nhanh (~10 – 15 phút trên GPU).\n",
                "3. **Tối ưu hóa ngưỡng Threshold:** Quét tìm ngưỡng F1 tối ưu trên tập validation và đánh giá độc lập trên tập test.\n",
                "4. **Đầu ra tại `/kaggle/working/`:**\n",
                "   - `best_checkpoint.pt`: Trọng số mô hình tốt nhất theo validation F1.\n",
                "   - `training_summary.json`: Lịch sử loss và F1 qua từng epoch.\n",
                "   - `test_metrics.json`: Bảng chỉ số hoàn chỉnh trên tập test (F1, Precision, Recall, Accuracy, ROC-AUC).\n",
                "   - `exp-bcb-004_astnn_results.tar.gz`: Gói kết quả đầy đủ.\n",
                "\n",
                "---"
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
                "import shutil\n",
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
                "sh(\"ls -la configs/exp004_astnn_bcb.yaml\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-install-deps-gpu",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT THƯ VIỆN & XÁC THỰC GPU\n",
                "sh(\"pip install -q 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn'\")\n",
                "\n",
                "import torch\n",
                "assert torch.cuda.is_available(), \"Cần bật GPU trong Settings -> Accelerator -> GPU (T4/P100)!\"\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)\n",
                "print(f\"GPU SẴN SÀNG: {gpu_name} | VRAM: {vram_gb} GB\")\n",
                "sh(\"nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-link-ast-cache",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT VÀ XÁC THỰC AST CACHE TỪ DATASET ĐÍNH KÈM\n",
                "ast_cache_found = None\n",
                "candidates = [\n",
                "    Path(\"/kaggle/input/semanticcodeclonedetection-dataset-m4/bcb_ast_cache.npz\"),\n",
                "    Path(\"/kaggle/input/bigclonebench-features/bcb_ast_cache.npz\"),\n",
                "]\n",
                "for c in candidates:\n",
                "    if c.exists():\n",
                "        ast_cache_found = c\n",
                "        break\n",
                "\n",
                "assert ast_cache_found is not None, f\"Không tìm thấy bcb_ast_cache.npz trong các dataset đính kèm! Đã tìm: {candidates}\"\n",
                "print(f\"Đã tìm thấy AST Cache: {ast_cache_found} ({ast_cache_found.stat().st_size / (1024*1024):.2f} MiB)\")\n",
                "\n",
                "# Tạo thư mục đích và liên kết cache\n",
                "target_dir = Path(\"artifacts/bcb_features\")\n",
                "target_dir.mkdir(parents=True, exist_ok=True)\n",
                "sh(f\"cp {ast_cache_found} {target_dir / 'bcb_ast_cache.npz'}\")\n",
                "\n",
                "# Kiểm tra nạp thử cache vào bộ nhớ\n",
                "from src.representations.ast.store import ASTTreeStore\n",
                "store = ASTTreeStore.build({}, cache_path=target_dir / \"bcb_ast_cache.npz\", verbose=True)\n",
                "print(f\"Xác thực AST cache: đã nạp thành công {len(store):,} cây hàm Java vào RAM!\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-prepare-subsets",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. TẢI CÁC SPLIT DỮ LIỆU ĐỂ HUẤN LUYỆN (TRAIN 50K, VAL 5K, TEST 5K)\n",
                "# Tải qua streaming cực nhanh (~30 giây)\n",
                "data_raw = Path(\"data/raw/bigclonebench\")\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(\"Đang tải tập dữ liệu phân chia chuẩn hóa cho BigCloneBench...\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split train --max-samples 50000\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split validation --max-samples 5000\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split test --max-samples 5000\")\n",
                "\n",
                "sh(\"ls -lh data/raw/bigclonebench/\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-preflight-diagnostic",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. KIỂM TRA PREFLIGHT & DIAGNOSTIC TRÊN GPU\n",
                "print(\"--- BẮT ĐẦU PREFLIGHT CHECK ---\")\n",
                "sh(\"python -m src.experiments.bcb_runner --config configs/exp004_astnn_bcb.yaml --mode preflight --device cuda\")\n",
                "\n",
                "print(\"\\n--- BẮT ĐẦU DIAGNOSTIC CHECK (XÁC NHẬN MÔ HÌNH HỌC ĐƯỢC TRÊN GPU) ---\")\n",
                "sh(\"python -m src.experiments.bcb_runner --config configs/exp004_astnn_bcb.yaml --mode diagnostic --device cuda\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-train-m3",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. TIẾN HÀNH HUẤN LUYỆN M3 (ASTNN) TRÊN GPU\n",
                "print(\"=\" * 60)\n",
                "print(\"BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC M3 (ASTNN)...\")\n",
                "print(\"=\" * 60)\n",
                "\n",
                "# Giới hạn 9h an toàn\n",
                "TIME_BUDGET_S = 9 * 3600\n",
                "sh(f\"python -m src.experiments.bcb_runner --config configs/exp004_astnn_bcb.yaml --mode train --device cuda --time-budget-s {TIME_BUDGET_S}\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c8-evaluate-m3",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. ĐÁNH GIÁ MÔ HÌNH TỐT NHẤT TRÊN TẬP TEST\n",
                "print(\"=\" * 60)\n",
                "print(\"BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP TEST...\")\n",
                "print(\"=\" * 60)\n",
                "sh(\"python -m src.experiments.bcb_runner --config configs/exp004_astnn_bcb.yaml --mode evaluate --device cuda\")"
            ],
        },
        {
            "cell_type": "code",
            "id": "c9-package-results",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 9. ĐÓNG GÓI KẾT QUẢ VÀO /kaggle/working/\n",
                "res_dir = Path(\"results/exp_bcb_004\")\n",
                "dest_dir = Path(\"/kaggle/working/exp_bcb_004_astnn_results\")\n",
                "dest_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "if res_dir.exists():\n",
                "    for f in res_dir.glob(\"*\"):\n",
                "        sh(f\"cp -r {f} {dest_dir}/\")\n",
                "        if f.suffix in (\".json\", \".pt\"):\n",
                "            sh(f\"cp {f} /kaggle/working/\")  # Sao chép trực tiếp ra root của working để dễ tải\n",
                "\n",
                "TAR_RESULTS = Path(\"/kaggle/working/exp-bcb-004_astnn_results.tar.gz\")\n",
                "sh(f\"tar -czf {TAR_RESULTS} results/\")\n",
                "\n",
                "print(\"\\n\" + \"=\" * 60)\n",
                "print(\"TỔNG KẾT KẾT QUẢ TẠI /kaggle/working/:\")\n",
                "print(\"=\" * 60)\n",
                "sh(\"ls -lh /kaggle/working/*.json /kaggle/working/*.tar.gz /kaggle/working/*.pt 2>/dev/null || ls -lh /kaggle/working/\")\n",
                "\n",
                "if (res_dir / \"test_metrics.json\").exists():\n",
                "    metrics = json.loads((res_dir / \"test_metrics.json\").read_text(encoding=\"utf-8\"))\n",
                "    print(\"\\nBẢNG KẾT QUẢ TEST METRICS CỦA M3 (ASTNN):\\n\", json.dumps(metrics, indent=2))\n",
                "\n",
                "print(\"\\nHUẤN LUYỆN VÀ ĐÁNH GIÁ M3 (ASTNN) HOÀN TẤT THÀNH CÔNG!\")"
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
    create_m3_nb()
