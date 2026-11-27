"""Tạo notebook Kaggle chạy A1->A2->A3 với M1 DiffMul gốc + M2 B5B6 (focal+aug+SupCon).

Checkpoint lấy từ 2 dataset Kaggle:
  - tranthikhanhlinh1709/diffmul-postprocess-ckpts  (m1_best_checkpoint.pt)
  - tranthikhanhlinh1709/diffmul-m2-b5b6-ckpt      (m2_b5b6_best_checkpoint.pt)
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "tranthikhanhlinh1709/bcb-diffmul-postprocess-m1-m2-b5b6-v2"
KERNEL_TITLE = "BCB DiffMul Postprocess M1 M2 B5B6 v2"
NOTEBOOK_NAME = "bcb_diffmul_postprocess_m1_m2b5b6.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "postprocess_m1_m2b5b6"


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
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-postprocess",
            "metadata": {},
            "source": [
                "# A1 → A2 → A3: M1 DiffMul gốc + M2 B5B6 (focal + augmentation + SupCon)\n",
                "\n",
                "- M1: CodeBERT DiffMul gốc (best checkpoint).\n",
                "- M2: GraphCodeBERT DiffMul B5/B6 (best checkpoint).\n",
                "- A1: calibration isotonic + ngưỡng 5-fold OOF trên validation.\n",
                "- A2: SWA best+last (bỏ qua nếu thiếu last).\n",
                "- A3: ensemble M1 + M2, trọng số quét trên OOF validation.\n",
                "\n",
                "Checkpoint lấy từ 2 dataset Kaggle; parquet val/test từ dataset hoặc tải từ HF.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c1-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. SETUP + STREAM LOG\n",
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time\n",
                "from pathlib import Path\n",
                "\n",
                "WORK = Path('/kaggle/working')\n",
                "CLEAN = WORK / 'clean_postprocess'\n",
                "CLEAN.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def sh(cmd):\n",
                "    print(f'\\n$ {cmd}', flush=True)\n",
                "    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "    log_file = WORK / 'live_progress.log'\n",
                "    with log_file.open('a', encoding='utf-8') as lf:\n",
                "        for line in iter(proc.stdout.readline, ''):\n",
                "            print(line, end='', flush=True)\n",
                "            lf.write(line)\n",
                "            lf.flush()\n",
                "    proc.stdout.close()\n",
                "    rc = proc.wait()\n",
                "    if rc != 0:\n",
                "        raise RuntimeError(f'Lenh that bai (rc={rc}): {cmd}')\n",
                "\n",
                "print('Python:', sys.version.split()[0], flush=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIAI NEN MA NGUON\n",
                f"payload = '{payload_b64}'\n",
                "repo_root = WORK / 'project'\n",
                "repo_root.mkdir(parents=True, exist_ok=True)\n",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as tar:\n",
                "    tar.extractall(repo_root)\n",
                "os.chdir(repo_root)\n",
                "print('Repo:', os.getcwd(), flush=True)\n",
                "sh('ls scripts/postprocess_diffmul.py src/postprocess/pipeline.py')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-gpu",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. GPU + DEPS\n",
                "import torch\n",
                "print('CUDA:', torch.cuda.is_available(), flush=True)\n",
                "if not torch.cuda.is_available():\n",
                "    raise SystemError('Kernel chua bat GPU!')\n",
                "print('GPU:', torch.cuda.get_device_name(0), flush=True)\n",
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-data-ckpt",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LINK PARQUET VAL/TEST VA CHECKPOINT TUNG DATASET KAGGLE\n",
                "input_path = Path('/kaggle/input')\n",
                "print('Inputs:', [p.name for p in input_path.iterdir()], flush=True)\n",
                "\n",
                "data_raw = repo_root / 'data/raw/bigclonebench'\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "for name in ('bcb_validation.parquet', 'bcb_test.parquet'):\n",
                "    cands = list(input_path.rglob(name))\n",
                "    dest = data_raw / name\n",
                "    if cands and not dest.exists():\n",
                "        sh(f'cp \"{cands[0]}\" \"{dest}\"')\n",
                "    print(name, 'exists=', dest.exists(), flush=True)\n",
                "if any(not (data_raw / n).exists() for n in ('bcb_validation.parquet', 'bcb_test.parquet')):\n",
                "    print('Tai lai splits tu Hugging Face...', flush=True)\n",
                "    sh('python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000')\n",
                "assert (data_raw / 'bcb_validation.parquet').exists(), 'Thieu bcb_validation.parquet!'\n",
                "assert (data_raw / 'bcb_test.parquet').exists(), 'Thieu bcb_test.parquet!'\n",
                "\n",
                "# Checkpoint tu dataset (tim theo ten file)\n",
                "ckpt_map = {\n",
                "    'm1_best_checkpoint.pt': 'results/exp_m1_diffmul/best_checkpoint.pt',\n",
                "    'm2_b5b6_best_checkpoint.pt': 'results/exp_m2_b5b6/best_checkpoint.pt',\n",
                "}\n",
                "found = {}\n",
                "for f in input_path.rglob('*.pt'):\n",
                "    if f.name in ckpt_map and f.name not in found:\n",
                "        found[f.name] = f\n",
                "        print('Dataset ckpt:', f.name, f.parent.parent.name, flush=True)\n",
                "\n",
                "for name, dest in ckpt_map.items():\n",
                "    dest_path = repo_root / dest\n",
                "    dest_path.parent.mkdir(parents=True, exist_ok=True)\n",
                "    if name in found and not dest_path.exists():\n",
                "        shutil.copy2(found[name], dest_path)\n",
                "    p = Path(dest)\n",
                "    print(dest, 'exists=', p.exists(), f', {p.stat().st_size/1e6:.1f} MB' if p.exists() else '', flush=True)\n",
                "    assert p.exists(), f'Thieu checkpoint {dest}'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-run",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. CHAY A1 -> A2 -> A3\n",
                "print('=' * 70)\n",
                "print('>>> BAT DAU POSTPROCESS M1 (goc) + M2 (B5B6) <<<')\n",
                "print('=' * 70)\n",
                "cmd = (\n",
                "    'python scripts/postprocess_diffmul.py '\n",
                "    '--val-file data/raw/bigclonebench/bcb_validation.parquet '\n",
                "    '--test-file data/raw/bigclonebench/bcb_test.parquet '\n",
                "    '--out-dir results/exp_diffmul_postprocess_m1_m2b5b6 '\n",
                "    '--batch-size 32 --precision fp16 --n-folds 5 --seed 42 '\n",
                "    '--m1-best results/exp_m1_diffmul/best_checkpoint.pt '\n",
                "    '--m2-best results/exp_m2_b5b6/best_checkpoint.pt'\n",
                ")\n",
                "t0 = time.perf_counter()\n",
                "sh(cmd)\n",
                "print(f'>>> HOAN TAT TRONG {(time.perf_counter()-t0)/60:.1f} PHUT!', flush=True)\n",
                "print((repo_root / 'results/exp_diffmul_postprocess_m1_m2b5b6/postprocess_report.md').read_text(encoding='utf-8'), flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-collect",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. THU THAP KET QUA + AUTO-CLEANUP\n",
                "WORK = Path('/kaggle/working')\n",
                "CLEAN = WORK / 'clean_postprocess'\n",
                "CLEAN.mkdir(parents=True, exist_ok=True)\n",
                "out_dir = repo_root / 'results' / 'exp_diffmul_postprocess_m1_m2b5b6'\n",
                "\n",
                "for p in sorted(out_dir.rglob('*')):\n",
                "    if p.is_file() and p.suffix in ('.json', '.md', '.npy', '.log'):\n",
                "        name = 'postprocess_' + p.name\n",
                "        shutil.copy2(p, CLEAN / name)\n",
                "        shutil.copy2(p, WORK / name)\n",
                "        print('  [SAVED]', name, flush=True)\n",
                "\n",
                "log_file = WORK / 'live_progress.log'\n",
                "if log_file.exists():\n",
                "    shutil.copy2(log_file, CLEAN / 'postprocess_live_progress.log')\n",
                "    shutil.copy2(log_file, WORK / 'postprocess_live_progress.log')\n",
                "\n",
                "with tarfile.open(WORK / 'diffmul_postprocess_m1_m2b5b6_bundle.tar.gz', 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "print('>>> Bundle:', WORK / 'diffmul_postprocess_m1_m2b5b6_bundle.tar.gz', flush=True)\n",
                "\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'data', 'artifacts', 'clean_postprocess'):\n",
                "    shutil.rmtree(WORK / junk, ignore_errors=True)\n",
                "print('DANH SACH OUTPUT:', flush=True)\n",
                "for p in sorted(WORK.glob('postprocess_*')):\n",
                "    print(f'  {p.name} ({p.stat().st_size/1e6:.2f} MB)', flush=True)\n",
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
        "dataset_sources": [
            "kichirou/semanticcodeclonedetection-dataset-m4",
            "tranthikhanhlinh1709/diffmul-postprocess-ckpts",
            "tranthikhanhlinh1709/diffmul-m2-b5b6-ckpt",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Đã tạo {DEPLOY_DIR / NOTEBOOK_NAME} | kernel {KERNEL_ID}")


if __name__ == "__main__":
    main()
