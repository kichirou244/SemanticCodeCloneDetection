"""Tạo notebook Kaggle huấn luyện M1/M2 DiffMul với B5 (focal) + B6 (augment + SupCon)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "tranthikhanhlinh1709/bcb-b5-b6-focal-aug-supcon-m1-m2"
KERNEL_TITLE = "BCB B5 B6 Focal Aug SupCon M1 M2"
NOTEBOOK_NAME = "bcb_b5b6_kaggle.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "b5b6"


def build_repo_payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("src", "scripts", "configs"):
            tar.add(
                ROOT / folder,
                arcname=folder,
                filter=lambda ti: None if "__pycache__" in ti.name else ti,
            )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_notebook(payload_b64: str) -> dict:
    def train_cell(model: str, cfg: str, backup_prefix: str, cell_id: str) -> dict:
        return {
            "cell_type": "code",
            "id": cell_id,
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# HUẤN LUYỆN {model} VỚI B5 + B6\n",
                "print('=' * 70)\n",
                f"print('>>> BAT DAU {model}: focal + augmentation + SupCon <<<')\n",
                "print('=' * 70)\n",
                f"cfg = '{cfg}'\n",
                "t_start = time.perf_counter()\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode preflight --device cuda')\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode train --device cuda --time-budget-s 18000')\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode evaluate --device cuda')\n",
                f"print(f'>>> HOAN TAT {model} TRONG {{(time.perf_counter()-t_start)/60:.1f}} PHUT!', flush=True)\n",
                "\n",
                "# Backup ngay kết quả trước khi chạy model tiếp theo\n",
                f"res_dir = repo_root / 'results' / '{backup_prefix}'\n",
                "if res_dir.exists():\n",
                "    for p in res_dir.glob('*'):\n",
                "        if p.is_file() and p.suffix in ('.pt', '.json'):\n",
                f"            name = '{backup_prefix}_' + p.name\n",
                "            shutil.copy2(p, CLEAN / name)\n",
                "            shutil.copy2(p, WORK / name)\n",
                "            print(f'  [SAVED] {name}', flush=True)\n",
                "else:\n",
                "    print('Canh bao: khong thay res_dir', flush=True)",
            ],
        }

    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-b5b6",
            "metadata": {},
            "source": [
                "# B5 + B6: Focal loss + Augmentation + SupCon cho M1/M2 DiffMul\n",
                "\n",
                "- **B5**: focal BCE (gamma=2, alpha=0.5) giảm áp lực easy negatives.\n",
                "- **B6**: xóa comment + đổi tên identifier nhất quán (giữ nhãn) + SupCon trên vector CLS.\n",
                "\n",
                "Huấn luyện lại 2 mô hình từ đầu với protocol giữ nguyên (30k train / 3k val / 5k test).\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c1-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time\n",
                "from pathlib import Path\n",
                "\n",
                "WORK = Path('/kaggle/working')\n",
                "CLEAN = WORK / 'clean_b5b6'\n",
                "CLEAN.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def sh(cmd):\n",
                "    print(f'\\n$ {cmd}', flush=True)\n",
                "    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "    with (WORK / 'live_progress.log').open('a', encoding='utf-8') as lf:\n",
                "        for line in iter(proc.stdout.readline, ''):\n",
                "            print(line, end='', flush=True)\n",
                "            lf.write(line)\n",
                "            lf.flush()\n",
                "    proc.stdout.close()\n",
                "    rc = proc.wait()\n",
                "    if rc != 0:\n",
                "        raise RuntimeError(f'Lenh that bai (rc={rc}): {cmd}')\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"payload = '{payload_b64}'\n",
                "repo_root = WORK / 'project'\n",
                "repo_root.mkdir(parents=True, exist_ok=True)\n",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as tar:\n",
                "    tar.extractall(repo_root)\n",
                "os.chdir(repo_root)\n",
                "print('Repo:', os.getcwd(), flush=True)\n",
                "sh('ls configs/tuning/exp_m1_b5b6.yaml configs/tuning/exp_m2_b5b6.yaml')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-gpu",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "print('CUDA:', torch.cuda.is_available(), flush=True)\n",
                "if not torch.cuda.is_available():\n",
                "    raise SystemError('Kernel chua bat GPU!')\n",
                "print('GPU:', torch.cuda.get_device_name(0), flush=True)\n",
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow \\\"tree-sitter>=0.23.0\\\" \\\"tree-sitter-java>=0.23.5\\\"')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_raw = repo_root / 'data/raw/bigclonebench'\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "for name in ('bcb_train.parquet', 'bcb_validation.parquet', 'bcb_test.parquet'):\n",
                "    cands = list(Path('/kaggle/input').rglob(name))\n",
                "    if cands and not (data_raw / name).exists():\n",
                "        sh(f'cp \"{cands[0]}\" \"{data_raw / name}\"')\n",
                "    print(name, 'exists=', (data_raw / name).exists(), flush=True)\n",
                "if any(not (data_raw / n).exists() for n in ('bcb_train.parquet', 'bcb_validation.parquet', 'bcb_test.parquet')):\n",
                "    print('Tai lai splits tu Hugging Face...', flush=True)\n",
                "    sh('python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000')\n",
                "for n in ('bcb_train.parquet', 'bcb_validation.parquet', 'bcb_test.parquet'):\n",
                "    assert (data_raw / n).exists(), f'Thieu {n}'",
            ],
        },
        train_cell("M1", "configs/tuning/exp_m1_b5b6.yaml", "exp_m1_b5b6", "c5-train-m1"),
        train_cell("M2", "configs/tuning/exp_m2_b5b6.yaml", "exp_m2_b5b6", "c6-train-m2"),
        {
            "cell_type": "code",
            "id": "c7-collect",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=' * 70)\n",
                "print('KET QUA TEST:', flush=True)\n",
                "for name in ('exp_m1_b5b6', 'exp_m2_b5b6'):\n",
                "    p = repo_root / 'results' / name / 'test_metrics.json'\n",
                "    print(name, p.read_text(encoding='utf-8') if p.exists() else 'MISSING', flush=True)\n",
                "\n",
                "with tarfile.open(WORK / 'b5b6_bundle.tar.gz', 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "print('>>> Bundle san sang.', flush=True)\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'clean_b5b6'):\n",
                "    shutil.rmtree(WORK / junk, ignore_errors=True)\n",
                "print('>>> XONG 100%!', flush=True)",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
            "accelerator": "gpu",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    payload = build_repo_payload()
    print(f"Payload: {len(payload)/1024:.1f} KB base64")
    nb = make_notebook(payload)
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / NOTEBOOK_NAME).write_text(text, encoding="utf-8")
    (ROOT / "notebooks" / NOTEBOOK_NAME).write_text(text, encoding="utf-8")
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": ["kichirou/semanticcodeclonedetection-dataset-m4"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Đã tạo {DEPLOY_DIR / NOTEBOOK_NAME} | kernel {KERNEL_ID}")


if __name__ == "__main__":
    main()
