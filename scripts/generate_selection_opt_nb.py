"""Tạo notebook Kaggle: tối ưu selection trên validation 10k + stacking."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "tranthikhanhlinh1709/bcb-selection-opt-val10k-stacking-v3"
KERNEL_TITLE = "BCB Selection Opt Val10k Stacking v3"
NOTEBOOK_NAME = "bcb_selection_opt_val10k.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "selection_opt"


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
            "id": "md-title",
            "metadata": {},
            "source": [
                "# Tối ưu protocol chọn ngưỡng/trọng số trên validation 10k + stacking\n",
                "\n",
                "- Validation mở rộng: 5.000 clone + 5.000 non-clone từ `validation_shard_0.parquet`.\n",
                "- Model: M1 DiffMul gốc + M2 B5/B6 (best checkpoint).\n",
                "- OOF 10-fold: isotonic calibration, ensemble weight grid, logistic stacking.\n",
                "- Ngưỡng/trọng số chốt trên OOF; test chỉ đánh giá cuối kèm bootstrap CI + McNemar.\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c1-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, io, os, shutil, subprocess, sys, tarfile, time\n",
                "from pathlib import Path\n",
                "WORK = Path('/kaggle/working')\n",
                "CLEAN = WORK / 'clean_sel'\n",
                "CLEAN.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "def sh(cmd):\n",
                "    print(f'\\n$ {cmd}', flush=True)\n",
                "    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
                "    with (WORK / 'live_progress.log').open('a', encoding='utf-8') as lf:\n",
                "        for line in iter(p.stdout.readline, ''):\n",
                "            print(line, end='', flush=True)\n",
                "            lf.write(line); lf.flush()\n",
                "    p.stdout.close()\n",
                "    rc = p.wait()\n",
                "    if rc != 0: raise RuntimeError(f'Fail rc={rc}: {cmd}')\n",
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
                "if not torch.cuda.is_available(): raise SystemError('Chua bat GPU!')\n",
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow')\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-link",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "input_path = Path('/kaggle/input')\n",
                "print('Inputs:', [p.name for p in input_path.iterdir()], flush=True)\n",
                "data_raw = repo_root / 'data/raw/bigclonebench'\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# validation shard + test parquet\n",
                "for src_name, dest_name in (('validation_shard_0.parquet', 'validation_shard_0.parquet'), ('bcb_test.parquet', 'bcb_test.parquet')):\n",
                "    cands = list(input_path.rglob(src_name))\n",
                "    dest = data_raw / dest_name\n",
                "    if cands and not dest.exists():\n",
                "        sh(f'cp \"{cands[0]}\" \"{dest}\"')\n",
                "    print(dest_name, dest.exists(), flush=True)\n",
                "    assert dest.exists(), f'Thieu {dest_name}'\n",
                "\n",
                "# checkpoint tu 2 dataset\n",
                "ckpt_map = {\n",
                "    'm1_best_checkpoint.pt': 'results/exp_m1_diffmul/best_checkpoint.pt',\n",
                "    'm2_b5b6_best_checkpoint.pt': 'results/exp_m2_b5b6/best_checkpoint.pt',\n",
                "}\n",
                "found = {}\n",
                "for f in input_path.rglob('*.pt'):\n",
                "    if f.name in ckpt_map and f.name not in found: found[f.name] = f\n",
                "for name, dest in ckpt_map.items():\n",
                "    p = repo_root / dest\n",
                "    p.parent.mkdir(parents=True, exist_ok=True)\n",
                "    if name in found and not p.exists(): shutil.copy2(found[name], p)\n",
                "    print(dest, p.exists(), flush=True)\n",
                "    assert p.exists(), f'Thieu {dest}'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-run",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('=' * 70)\n",
                "print('>>> BAT DAU OPTIMIZE SELECTION <<<')\n",
                "print('=' * 70)\n",
                "t0 = time.perf_counter()\n",
                "sh('python scripts/optimize_selection.py --n-val 10000 --n-folds 10 --batch-size 32')\n",
                "print(f'>>> XONG TRONG {(time.perf_counter()-t0)/60:.1f} PHUT', flush=True)\n",
                "print((repo_root / 'results/exp_selection_opt/report.md').read_text(encoding='utf-8'), flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-collect",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "WORK = Path('/kaggle/working')\n",
                "out_dir = repo_root / 'results/exp_selection_opt'\n",
                "for p in sorted(out_dir.rglob('*')):\n",
                "    if p.is_file() and p.suffix in ('.json', '.md', '.npy', '.log'):\n",
                "        name = 'sel_' + p.name\n",
                "        shutil.copy2(p, CLEAN / name)\n",
                "        shutil.copy2(p, WORK / name)\n",
                "        print('  [SAVED]', name, flush=True)\n",
                "log = WORK / 'live_progress.log'\n",
                "if log.exists():\n",
                "    shutil.copy2(log, CLEAN / 'sel_live_progress.log')\n",
                "    shutil.copy2(log, WORK / 'sel_live_progress.log')\n",
                "with tarfile.open(WORK / 'selection_opt_bundle.tar.gz', 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'): tar.add(f, arcname=f.name)\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'data', 'clean_sel'):\n",
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
    print(f"Payload: {len(payload)/1024:.1f} KB")
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
            "tranthikhanhlinh1709/bcb-validation-shard0",
            "tranthikhanhlinh1709/diffmul-postprocess-ckpts",
            "tranthikhanhlinh1709/diffmul-m2-b5b6-ckpt",
            "tranthikhanhlinh1709/bcb-test-parquet",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {DEPLOY_DIR / NOTEBOOK_NAME} | {KERNEL_ID}")


if __name__ == "__main__":
    main()
