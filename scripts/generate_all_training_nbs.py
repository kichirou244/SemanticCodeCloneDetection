"""Tạo 4 notebook huấn luyện tự chứa (Self-contained) cho M1, M2, M3, M4 trên Kaggle GPU."""

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


def make_notebook(model_id: str, model_name: str, config_file: str, payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": f"md-title-{model_id.lower()}",
            "metadata": {},
            "source": [
                f"# HUẤN LUYỆN {model_id} ({model_name}) TRÊN BIGCLONEBENCH JAVA (GPU)\n",
                "\n",
                f"Notebook này thực hiện huấn luyện mô hình **{model_id} ({model_name})** trên BigCloneBench Java sử dụng GPU Kaggle.\n",
                "\n",
                "### Thông tin mô hình:\n",
                f"- **Mã mô hình:** {model_id}\n",
                f"- **Tên phương pháp:** {model_name}\n",
                f"- **File cấu hình:** `{config_file}`\n",
                "- **Dataset liên kết:** `semanticcodeclonedetection-dataset-m4`\n",
                "- **Đầu ra tại `/kaggle/working/`:** `best_checkpoint.pt`, `training_summary.json`, `test_metrics.json`\n",
                "\n",
                "---",
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
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time\n",
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
                "print(\"Python:\", sys.version.split()[0])",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIẢI NÉN MÃ NGUỒN DỰ ÁN MỚI NHẤT (SELF-CONTAINED)\n",
                "WORK_DIR = Path(\"/kaggle/working/project\")\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                f"BUNDLE_B64 = {json.dumps(payload_b64)}\n",
                "\n",
                "print(f\"Giải nén mã nguồn ({len(BUNDLE_B64)/1024:.1f} KB) vào {WORK_DIR}...\")\n",
                "buf = io.BytesIO(base64.b64decode(BUNDLE_B64))\n",
                "with tarfile.open(fileobj=buf, mode=\"r:gz\") as tar:\n",
                "    tar.extractall(WORK_DIR)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                f"sh(\"ls -la {config_file}\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-deps-gpu",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT THƯ VIỆN & XÁC THỰC GPU\n",
                "sh(\"pip install -q 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'transformers>=4.30.0' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn'\")\n",
                "\n",
                "import torch\n",
                "assert torch.cuda.is_available(), \"Cần bật GPU trong Settings -> Accelerator -> GPU (T4/P100)!\"\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)\n",
                "print(f\"GPU SẴN SÀNG: {gpu_name} | VRAM: {vram_gb} GB\")\n",
                "sh(\"nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-dataset-features",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT ĐẶC TRƯNG TỪ DATASET NẾU CẦN (AST / CPG)\n",
                "in_dir = Path(\"/kaggle/input/semanticcodeclonedetection-dataset-m4\")\n",
                "if in_dir.exists():\n",
                "    print(f\"Dataset đính kèm: {in_dir}\")\n",
                "    sh(f\"ls -lh {in_dir}\")\n",
                "    # Nạp AST cache nếu là M3\n",
                "    ast_p = in_dir / \"bcb_ast_cache.npz\"\n",
                "    if ast_p.exists():\n",
                "        os.makedirs(\"artifacts/bcb_features\", exist_ok=True)\n",
                "        sh(f\"cp {ast_p} artifacts/bcb_features/\")\n",
                "    # Nạp CPG archive nếu là M4\n",
                "    cpg_tar = in_dir / \"bcb_cpg_export.tar.gz\"\n",
                "    if cpg_tar.exists():\n",
                "        os.makedirs(\"artifacts\", exist_ok=True)\n",
                "        print(\"Đang giải nén CPG cache...\")\n",
                "        sh(f\"tar -xzf {cpg_tar} -C artifacts/\")\n",
                "else:\n",
                "    print(\"Chưa phát hiện dataset đính kèm, sử dụng đặc trưng mặc định.\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-download-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. TẢI CÁC SPLIT DỮ LIỆU ĐỂ HUẤN LUYỆN (TRAIN 50K, VAL 5K, TEST 5K)\n",
                "data_raw = Path(\"data/raw/bigclonebench\")\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "print(\"Tải các phân tập dữ liệu BigCloneBench (Train: 50k, Val: 5k, Test: 5k)...\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split train --max-samples 50000\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split validation --max-samples 5000\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --split test --max-samples 5000\")\n",
                "sh(\"ls -lh data/raw/bigclonebench/\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-preflight-diagnostic",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. PREFLIGHT & DIAGNOSTIC CHECK\n",
                f"print(\"--- PREFLIGHT {model_id} ---\")\n",
                f"sh(\"python -m src.experiments.bcb_runner --config {config_file} --mode preflight --device cuda\")\n",
                f"print(\"\\n--- DIAGNOSTIC CHECK {model_id} ---\")\n",
                f"sh(\"python -m src.experiments.bcb_runner --config {config_file} --mode diagnostic --device cuda\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-train",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 7. TIẾN HÀNH HUẤN LUYỆN {model_id} ({model_name}) TRÊN GPU\n",
                "print(\"=\" * 60)\n",
                f"print(\"BẮT ĐẦU HUẤN LUYỆN CHÍNH THỨC {model_id} ({model_name})...\")\n",
                "print(\"=\" * 60)\n",
                "TIME_BUDGET_S = 9 * 3600\n",
                f"sh(f\"python -m src.experiments.bcb_runner --config {config_file} --mode train --device cuda --time-budget-s {{TIME_BUDGET_S}}\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c8-evaluate",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 8. ĐÁNH GIÁ MÔ HÌNH {model_id} TRÊN TẬP TEST\n",
                "print(\"=\" * 60)\n",
                f"print(\"ĐÁNH GIÁ MÔ HÌNH {model_id} TRÊN TẬP TEST...\")\n",
                "print(\"=\" * 60)\n",
                f"sh(\"python -m src.experiments.bcb_runner --config {config_file} --mode evaluate --device cuda\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c9-package",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 9. ĐÓNG GÓI KẾT QUẢ VÀO /kaggle/working/\n",
                "exp_id = json.load(open(f\"{WORK_DIR}/{config_file}\"))[\"experiment\"][\"id\"].lower().replace(\"-\", \"_\")\n",
                "res_dir = Path(\"results\") / exp_id\n",
                f"dest_dir = Path(\"/kaggle/working/{model_id.lower()}_results\")\n",
                "dest_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "if res_dir.exists():\n",
                "    for f in res_dir.glob(\"*\"):\n",
                "        sh(f\"cp -r {f} {dest_dir}/\")\n",
                "        if f.suffix in (\".json\", \".pt\"):\n",
                "            sh(f\"cp {f} /kaggle/working/\")\n",
                "\n",
                f"TAR_RESULTS = Path(\"/kaggle/working/{model_id.lower()}_results.tar.gz\")\n",
                "sh(f\"tar -czf {TAR_RESULTS} results/\")\n",
                "sh(\"ls -lh /kaggle/working/*.json /kaggle/working/*.tar.gz 2>/dev/null || ls -lh /kaggle/working/\")\n",
                "\n",
                "if (res_dir / \"test_metrics.json\").exists():\n",
                "    metrics = json.loads((res_dir / \"test_metrics.json\").read_text(encoding=\"utf-8\"))\n",
                f"    print(\"\\nBẢNG TEST METRICS CỦA {model_id}:\\n\", json.dumps(metrics, indent=2))\n",
                f"print(\"\\nHUẤN LUYỆN VÀ ĐÁNH GIÁ {model_id} HOÀN TẤT THÀNH CÔNG!\")",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def generate_all():
    payload = build_repo_payload()
    models = [
        ("M3", "ASTNN", "configs/exp004_astnn_bcb.yaml", "notebooks/bcb_exp-bcb-004_astnn_kaggle.ipynb"),
        ("M1", "CodeBERT", "configs/exp002_codebert_bcb.yaml", "notebooks/bcb_exp-bcb-002_codebert_kaggle.ipynb"),
        ("M2", "GraphCodeBERT", "configs/exp003_graphcodebert_bcb.yaml", "notebooks/bcb_exp-bcb-003_graphcodebert_kaggle.ipynb"),
        ("M4", "CPG-GNN", "configs/exp005_cpg_gnn_bcb.yaml", "notebooks/bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb"),
    ]
    for mid, mname, cfg, out_path in models:
        nb = make_notebook(mid, mname, cfg, payload)
        Path(out_path).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Đã sinh notebook {mid}: {out_path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    generate_all()
