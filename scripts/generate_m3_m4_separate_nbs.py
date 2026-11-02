"""Tạo 2 notebook riêng biệt cho M3 (ASTNN) và M4 (CPG-GNN) để huấn luyện độc lập trên Kaggle GPU."""

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


def make_single_model_notebook(model_id: str, model_name: str, config_file: str, payload_b64: str) -> dict:
    is_m3 = (model_id == "M3")
    is_m4 = (model_id == "M4")

    cells = [
        {
            "cell_type": "markdown",
            "id": f"md-title-{model_id.lower()}",
            "metadata": {},
            "source": [
                f"# HUẤN LUYỆN & ĐÁNH GIÁ {model_id} ({model_name}) TRÊN BIGCLONEBENCH JAVA (GPU)\n",
                "\n",
                f"Notebook này thực hiện huấn luyện độc lập mô hình **{model_id} ({model_name})** trên BigCloneBench Java.\n",
                "\n",
                "### Thông tin cấu hình:\n",
                f"- **Mã mô hình:** {model_id}\n",
                f"- **Tên phương pháp:** {model_name}\n",
                f"- **File cấu hình:** `{config_file}`\n",
                "- **Dataset liên kết:** `kichirou/semanticcodeclonedetection-dataset-m4`\n",
                "- **Splits:** 30.000 Train, 3.000 Val, 5.000 Test cân bằng 50:50\n",
                "- **Tự động lưu checkpoint & test metrics riêng biệt:** Kết quả lưu tại `/kaggle/working/`\n",
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
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ STREAM LOG\n",
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time, yaml\n",
                "from pathlib import Path\n",
                "\n",
                "def sh(cmd, **kw):\n",
                "    print(f\"\\n$ {cmd}\", flush=True)\n",
                "    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "    log_file = Path(\"/kaggle/working/live_progress.log\")\n",
                "    with log_file.open(\"a\", encoding=\"utf-8\") as lf:\n",
                "        for line in iter(proc.stdout.readline, ''):\n",
                "            print(line, end='', flush=True)\n",
                "            lf.write(line)\n",
                "            lf.flush()\n",
                "    proc.stdout.close()\n",
                "    rc = proc.wait()\n",
                "    if rc != 0:\n",
                "        raise RuntimeError(f\"Lệnh thất bại (rc={rc}): {cmd}\")\n",
                "    return \"\"\n",
                "\n",
                "print(\"Python:\", sys.version.split()[0], flush=True)",
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
                f"CONFIG_FILE = \"{config_file}\"\n",
                f"BUNDLE_B64 = {json.dumps(payload_b64)}\n",
                "\n",
                f"print(f\"Giải nén mã nguồn ({{len(BUNDLE_B64)/1024:.1f}} KB) vào {{WORK_DIR}}...\")\n",
                "buf = io.BytesIO(base64.b64decode(BUNDLE_B64))\n",
                "with tarfile.open(fileobj=buf, mode=\"r:gz\") as tar:\n",
                "    tar.extractall(WORK_DIR)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                "sh(f\"ls -la {CONFIG_FILE}\")",
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
                "sh(\"pip install -q 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'transformers>=4.30.0' 'torch_geometric' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn'\")\n",
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
            "id": "c4-dataset-link",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT ĐẶC TRƯNG TỪ DATASET NẾU CÓ\n",
                "import numpy as np\n",
                "ds_candidates = [\n",
                "    Path(\"/kaggle/input/datasets/kichirou/semanticcodeclonedetection-dataset-m4\"),\n",
                "    Path(\"/kaggle/input/semanticcodeclonedetection-dataset-m4\"),\n",
                "]\n",
                "ds_root = None\n",
                "for d in ds_candidates:\n",
                "    if d.exists():\n",
                "        ds_root = d\n",
                "        break\n",
                "\n",
                "print(f\"Thư mục dataset phát hiện: {ds_root}\")\n",
                "if ds_root:\n",
                "    sh(f\"ls -la {ds_root}/\")\n",
            ] + ([
                "# Liên kết AST cache cho M3\n",
                "os.makedirs(\"artifacts/bcb_features\", exist_ok=True)\n",
                "ast_target = Path(\"artifacts/bcb_features/bcb_ast_cache.npz\")\n",
                "ast_src = ds_root / \"bcb_ast_cache\" if ds_root else None\n",
                "if ast_src and ast_src.exists():\n",
                "    print(f\"Phát hiện thư mục AST cache tại: {ast_src}\")\n",
                "    npy_files = list(ast_src.glob(\"*.npy\"))\n",
                "    if npy_files:\n",
                "        print(f\"Tìm thấy {len(npy_files)} file .npy, tiến hành đóng gói {ast_target.name}...\")\n",
                "        arrays = {f.stem: np.load(f, allow_pickle=True) for f in npy_files}\n",
                "        np.savez_compressed(ast_target, **arrays)\n",
                "        print(f\"Đã tạo thành công {ast_target} ({ast_target.stat().st_size / 1024:.1f} KB)!\")\n",
                "elif (ds_root / \"bcb_ast_cache.npz\").exists() if ds_root else False:\n",
                "    sh(f\"cp {ds_root}/bcb_ast_cache.npz {ast_target}\")\n",
                "    print(f\"Đã sao chép {ast_target} từ dataset!\")\n",
                "else:\n",
                "    print(\"Thông báo: Chưa có pre-computed AST cache, sẽ compile online.\")\n",
            ] if is_m3 else []) + ([
                "# Liên kết CPG Graphs cho M4\n",
                "os.makedirs(\"artifacts\", exist_ok=True)\n",
                "cpg_src = ds_root / \"bcb_cpg_export\" if ds_root else None\n",
                "if cpg_src and cpg_src.exists():\n",
                "    print(f\"Phát hiện thư mục CPG export tại: {cpg_src}\")\n",
                "    inner = cpg_src / \"cpg_export\"\n",
                "    dest = Path(\"artifacts/cpg_export\")\n",
                "    if inner.exists():\n",
                "        sh(f\"cp -r {inner} artifacts/\")\n",
                "    else:\n",
                "        sh(f\"cp -r {cpg_src} artifacts/cpg_export\")\n",
                "    cpg_count = len(list(dest.glob(\"*.json\")))\n",
                "    print(f\"Đã liên kết {cpg_count:,} đồ thị CPG JSON vào {dest}!\")\n",
                "elif (ds_root / \"bcb_cpg_export.tar.gz\").exists() if ds_root else False:\n",
                "    sh(f\"tar -xzf {ds_root}/bcb_cpg_export.tar.gz -C artifacts/\")\n",
                "    print(\"Đã giải nén bcb_cpg_export.tar.gz vào artifacts/!\")\n",
                "else:\n",
                "    print(\"Cảnh báo: Không tìm thấy tài nguyên CPG export!\")\n",
            ] if is_m4 else []),
        },
        {
            "cell_type": "code",
            "id": "c5-download-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. TẢI CÁC SPLIT DỮ LIỆU ĐỂ HUẤN LUYỆN (TRAIN 30.000, VAL 3.000, TEST 5.000 CÂN BẰNG 50:50)\n",
                "data_raw = Path(\"data/raw/bigclonebench\")\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "print(\"Tải các phân tập dữ liệu BigCloneBench (Train: 30k, Val: 3k, Test: 5k) cân bằng 50:50...\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000\")\n",
                "sh(\"ls -lh data/raw/bigclonebench/\")\n",
                "assert (data_raw / 'bcb_train.parquet').exists() and (data_raw / 'bcb_train.parquet').stat().st_size > 0, 'bcb_train.parquet phai ton tai va co du lieu!'",
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
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode preflight --device cuda\")\n",
                f"print(\"\\n--- DIAGNOSTIC CHECK {model_id} ---\")\n",
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode diagnostic --device cuda\")",
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
                f"print(\"BẮT ĐẦU HUẤN LUYỆN {model_id} ({model_name})...\")\n",
                "print(\"=\" * 60)\n",
                "TIME_BUDGET_S = 9 * 3600\n",
                f"try:\n",
                f"    sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode train --device cuda --time-budget-s {{TIME_BUDGET_S}}\")\n",
                f"finally:\n",
                f"    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:\n",
                f"        exp_id = yaml.safe_load(f)['experiment']['id'].lower().replace('-', '_')\n",
                f"    res_dir = Path('results') / exp_id\n",
                f"    dest_dir = Path('/kaggle/working/{model_id.lower()}_results')\n",
                f"    dest_dir.mkdir(parents=True, exist_ok=True)\n",
                f"    if res_dir.exists():\n",
                f"        for f in res_dir.glob('*'):\n",
                f"            sh(f'cp -r {{f}} {{dest_dir}}/')\n",
                f"            if f.suffix in ('.json', '.pt'):\n",
                f"                sh(f'cp {{f}} /kaggle/working/')\n",
                f"    sh(f'tar -czf /kaggle/working/{model_id.lower()}_results.tar.gz results/')\n",
                f"    print(f'>>> ĐÃ SAO LƯU CHECKPOINT VÀ KẾT QUẢ HUẤN LUYỆN CỦA {model_id} VÀO /kaggle/working/')",
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
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode evaluate --device cuda\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c9-package",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 9. ĐÓNG GÓI & TỔNG KẾT KẾT QUẢ VÀO /kaggle/working/\n",
                "with open(CONFIG_FILE, 'r', encoding='utf-8') as f:\n",
                "    exp_id = yaml.safe_load(f)['experiment']['id'].lower().replace('-', '_')\n",
                "res_dir = Path('results') / exp_id\n",
                f"dest_dir = Path('/kaggle/working/{model_id.lower()}_results')\n",
                "dest_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "if res_dir.exists():\n",
                "    for f in res_dir.glob('*'):\n",
                "        sh(f'cp -r {f} {dest_dir}/')\n",
                "        if f.suffix in ('.json', '.pt'):\n",
                "            sh(f'cp {f} /kaggle/working/')\n",
                "\n",
                f"TAR_RESULTS = Path('/kaggle/working/{model_id.lower()}_results.tar.gz')\n",
                "sh(f'tar -czf {TAR_RESULTS} results/')\n",
                "sh('ls -lh /kaggle/working/*.json /kaggle/working/*.tar.gz 2>/dev/null || ls -lh /kaggle/working/')\n",
                "\n",
                "if (res_dir / 'test_metrics.json').exists():\n",
                "    metrics = json.loads((res_dir / 'test_metrics.json').read_text(encoding='utf-8'))\n",
                f"    print('\\nBẢNG TEST METRICS CỦA {model_id}:\\n', json.dumps(metrics, indent=2))\n",
                f"print('\\nHUẤN LUYỆN VÀ ĐÁNH GIÁ {model_id} HOÀN TẤT THÀNH CÔNG!')",
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


def main():
    payload = build_repo_payload()
    
    # 1. M3 Notebook
    nb_m3 = make_single_model_notebook("M3", "ASTNN", "configs/exp004_astnn_bcb.yaml", payload)
    m3_dir = Path("kaggle_deploy/m3")
    m3_dir.mkdir(parents=True, exist_ok=True)
    (m3_dir / "bcb_exp-bcb-004_astnn_kaggle.ipynb").write_text(json.dumps(nb_m3, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("notebooks/bcb_exp-bcb-004_astnn_kaggle.ipynb").write_text(json.dumps(nb_m3, indent=1, ensure_ascii=False), encoding="utf-8")
    (m3_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "kichirou/bcb-m3-astnn-opt",
        "title": "BCB M3 ASTNN Opt",
        "code_file": "bcb_exp-bcb-004_astnn_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["kichirou/semanticcodeclonedetection-dataset-m4"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }, indent=2), encoding="utf-8")
    print("Đã tạo kaggle_deploy/m3/")

    # 2. M4 Notebook
    nb_m4 = make_single_model_notebook("M4", "CPG-GNN", "configs/exp005_cpg_gnn_bcb.yaml", payload)
    m4_dir = Path("kaggle_deploy/m4")
    m4_dir.mkdir(parents=True, exist_ok=True)
    (m4_dir / "bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb").write_text(json.dumps(nb_m4, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("notebooks/bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb").write_text(json.dumps(nb_m4, indent=1, ensure_ascii=False), encoding="utf-8")
    (m4_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "kichirou/bcb-m4-cpg-gnn-opt",
        "title": "BCB M4 CPG GNN Opt",
        "code_file": "bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["kichirou/semanticcodeclonedetection-dataset-m4"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }, indent=2), encoding="utf-8")
    print("Đã tạo kaggle_deploy/m4/")


if __name__ == "__main__":
    main()
