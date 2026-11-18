"""Tạo Notebook Kaggle cho thử nghiệm Pooling Contrastive (M1, M2) và True DFG (M2)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile


def build_repo_payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("src", arcname="src", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("scripts", arcname="scripts", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("configs", arcname="configs", filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_sota_pool_dfg_notebook(payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-pool-dfg",
            "metadata": {},
            "source": [
                "# HUẤN LUYỆN POOLING CONTRASTIVE (M1 & M2) VÀ TRUE DATA FLOW GRAPH (DFG) TRÊN 30.000 CẶP DỮ LIỆU\n",
                "\n",
                "Notebook này thực hiện huấn luyện 3 cấu hình đột phá cho CodeBERT và GraphCodeBERT:\n",
                "1. **M1_CodeBERT_Pool_SupCon:** Pooling Mean+Max+CLS kết hợp Supervised Contrastive Loss & Focal Loss.\n",
                "2. **M2_GraphCodeBERT_Pool_SupCon:** Pooling Mean+Max+CLS kết hợp Supervised Contrastive Loss & Focal Loss.\n",
                "3. **M2_GraphCodeBERT_True_DFG:** Khai thác luồng dữ liệu (Data Flow Graph) trích xuất bằng Tree-sitter Java và ma trận Graph-Guided Attention Mask.\n",
                "\n",
                "Đánh giá độc lập trên cùng 5.000 cặp BigCloneBench Test chuẩn 1:1.\n",
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
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ STREAM LOG AN TOÀN\n",
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time\n",
                "from pathlib import Path\n",
                "\n",
                "def sh(cmd, **kw):\n",
                "    print(f'\\n$ {cmd}', flush=True)\n",
                "    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "    log_file = Path('/kaggle/working/live_progress.log')\n",
                "    with log_file.open('a', encoding='utf-8') as lf:\n",
                "        for line in iter(proc.stdout.readline, ''):\n",
                "            print(line, end='', flush=True)\n",
                "            lf.write(line)\n",
                "            lf.flush()\n",
                "    proc.stdout.close()\n",
                "    rc = proc.wait()\n",
                "    if rc != 0:\n",
                "        raise RuntimeError(f'Lệnh thất bại (rc={rc}): {cmd}')\n",
                "    return ''\n",
                "\n",
                "print('Python:', sys.version.split()[0], flush=True)\n",
                "print('Thư mục làm việc hiện tại:', os.getcwd(), flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack-repo",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIẢI NÉN MÃ NGUỒN DỰ ÁN VÀO THƯ MỤC RIÊNG\n",
                "WORK_DIR = Path('/kaggle/working/project')\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                f"BUNDLE_B64 = {json.dumps(payload_b64)}\n",
                "\n",
                "print(f'Giải nén mã nguồn ({len(BUNDLE_B64)/1024:.1f} KB) vào {WORK_DIR}...')\n",
                "buf = io.BytesIO(base64.b64decode(BUNDLE_B64))\n",
                "with tarfile.open(fileobj=buf, mode='r:gz') as tar:\n",
                "    tar.extractall(WORK_DIR)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                "\n",
                "assert Path('src/experiments/train_sota_pool_dfg.py').exists(), 'Thiếu file train_sota_pool_dfg.py'\n",
                "print('Đã giải nén mã nguồn dự án thành công!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-install-deps",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT DEPENDENCIES & XÁC THỰC GPU BẮT BUỘC\n",
                "sh(\"pip install -q --no-input 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'transformers>=4.30.0' 'datasets>=2.14.0' 'pyarrow' 'tqdm' 'scikit-learn'\")\n",
                "\n",
                "import torch\n",
                "if not torch.cuda.is_available():\n",
                "    raise SystemError('GPU KHONG KHA DUNG! Hay chon GPU (T4/P100) truoc khi chay.')\n",
                "\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)\n",
                "print(f'>>> GPU HOAT DONG: {gpu_name} | VRAM: {vram_gb} GB <<<')\n",
                "sh('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-link-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT HOẶC TẢI DỮ LIỆU TỪ DATASET KAGGLE\n",
                "input_path = Path('/kaggle/input')\n",
                "os.makedirs('data/raw/bigclonebench', exist_ok=True)\n",
                "os.makedirs('results', exist_ok=True)\n",
                "\n",
                "for f in input_path.rglob('*.parquet'):\n",
                "    if 'train_shard_0' in f.name:\n",
                "        dest = Path('data/raw/bigclonebench/train_shard_0.parquet')\n",
                "        if not dest.exists():\n",
                "            shutil.copy(f, dest)\n",
                "            print(f'Đã nạp {f.name} ({dest.stat().st_size/1e6:.1f} MB)')\n",
                "    elif 'validation_shard_0' in f.name:\n",
                "        dest = Path('data/raw/bigclonebench/validation_shard_0.parquet')\n",
                "        if not dest.exists():\n",
                "            shutil.copy(f, dest)\n",
                "            print(f'Đã nạp {f.name} ({dest.stat().st_size/1e6:.1f} MB)')\n",
                "\n",
                "for f in input_path.rglob('bcb_5k_test_pairs.jsonl'):\n",
                "    dest = Path('results/bcb_5k_test_pairs.jsonl')\n",
                "    if not dest.exists():\n",
                "        shutil.copy(f, dest)\n",
                "        print(f'Đã nạp {f.name} ({dest.stat().st_size/1e6:.1f} MB)')\n",
                "\n",
                "if not Path('data/raw/bigclonebench/train_shard_0.parquet').exists() or not Path('results/bcb_5k_test_pairs.jsonl').exists():\n",
                "    print('Chạy script chuẩn bị dữ liệu...')\n",
                "    sh('python scripts/prepare_hybrid_level3_data.py')\n",
                "\n",
                "assert Path('data/raw/bigclonebench/train_shard_0.parquet').exists(), 'Thiếu train_shard_0.parquet!'\n",
                "assert Path('results/bcb_5k_test_pairs.jsonl').exists(), 'Thiếu bcb_5k_test_pairs.jsonl!'\n",
                "print('Dữ liệu đã sẵn sàng!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-run-training",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. HUẤN LUYỆN POOLING SUPCON & TRUE DFG GRAPHCODEBERT\n",
                "cmd = (\n",
                "    'python src/experiments/train_sota_pool_dfg.py '\n",
                "    '--epochs 4 '\n",
                "    '--batch_size 32 '\n",
                "    '--lr 2e-5 '\n",
                "    '--output_dir results/exp_sota_pool_dfg'\n",
                ")\n",
                "sh(cmd)\n",
                "\n",
                "# 6. SAO CHÉP KẾT QUẢ ĐẦU RA RA /kaggle/working\n",
                "print('Sao chép kết quả ra /kaggle/working...')\n",
                "sh('cp -r results/exp_sota_pool_dfg/* /kaggle/working/')\n",
                "sh('ls -la /kaggle/working/')\n",
                "print('>>> HOÀN THÀNH HUẤN LUYỆN & KIỂM THỬ POOLING SUPCON & TRUE DFG! <<<')",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
            "accelerator": "gpu",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    deploy_dir = Path("kaggle_deploy/sota_pool_dfg")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Nén mã nguồn repo vào base64 payload...")
    payload_b64 = build_repo_payload()
    print(f"Kích thước payload: {len(payload_b64) / 1024:.2f} KB")

    nb = make_sota_pool_dfg_notebook(payload_b64)
    nb_path = deploy_dir / "bcb_sota_pool_dfg_kaggle.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Đã tạo Notebook: {nb_path} ({nb_path.stat().st_size / 1024:.2f} KB)")

    metadata = {
        "id": "tranthikhanhlinh1709/bcb-sota-pool-dfg-30k",
        "title": "BCB SOTA Pool DFG 30k",
        "code_file": "bcb_sota_pool_dfg_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "kichirou/semanticcodeclonedetection-dataset-m4"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }

    meta_path = deploy_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Đã tạo Metadata: {meta_path}")
    print(">>> Hoàn tất tạo gói triển khai SOTA Pool DFG!")


if __name__ == "__main__":
    main()
