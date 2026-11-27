"""Tạo 2 notebook Kaggle riêng: M1 DiffMul+SupCon và M2 DiffMul+SupCon (ablation thuần)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def build_repo_payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("src", "scripts", "configs"):
            tar.add(ROOT / folder, arcname=folder, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_notebook(payload_b64: str, model: str, cfg: str, exp_id: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title",
            "metadata": {},
            "source": [
                f"# Ablation thuần SupCon — {model} DiffMul\n",
                "\n",
                f"Chỉ thêm SupCon vào {model} DiffMul gốc; giữ nguyên LR/WD/dropout/epoch/loss/augmentation.\n",
                "\n",
                f"Config: `{cfg}`. Baseline cần so sánh: {model} DiffMul gốc.\n",
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
                "CLEAN = WORK / 'clean_ab'\n",
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
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow \\\"tree-sitter>=0.23.0\\\" \\\"tree-sitter-java>=0.23.5\\\"')\n",
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
                "    dest = data_raw / name\n",
                "    if cands and not dest.exists(): sh(f'cp \"{cands[0]}\" \"{dest}\"')\n",
                "    print(name, dest.exists(), flush=True)\n",
                "    assert dest.exists(), f'Thieu {name}'\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-train",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"cfg = '{cfg}'\n",
                f"print('>>> BAT DAU {exp_id} <<<')\n",
                "t0 = time.perf_counter()\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode preflight --device cuda')\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode train --device cuda --time-budget-s 18000')\n",
                "sh(f'python -m src.experiments.bcb_runner --config {cfg} --mode evaluate --device cuda')\n",
                "print(f'>>> XONG TRONG {(time.perf_counter()-t0)/60:.1f} PHUT', flush=True)\n",
                f"print((repo_root / 'results/{exp_id}/test_metrics.json').read_text(encoding='utf-8'), flush=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-collect",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"res_dir = repo_root / 'results/{exp_id}'\n",
                "for p in res_dir.glob('*'):\n",
                "    if p.is_file() and p.suffix in ('.pt', '.json'):\n",
                f"        name = '{exp_id}_' + p.name\n",
                "        shutil.copy2(p, CLEAN / name)\n",
                "        shutil.copy2(p, WORK / name)\n",
                "        print('  [SAVED]', name, flush=True)\n",
                "log = WORK / 'live_progress.log'\n",
                "if log.exists():\n",
                f"    shutil.copy2(log, CLEAN / '{exp_id}_live_progress.log')\n",
                f"    shutil.copy2(log, WORK / '{exp_id}_live_progress.log')\n",
                f"with tarfile.open(WORK / '{exp_id}_bundle.tar.gz', 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'): tar.add(f, arcname=f.name)\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'data', 'clean_ab'):\n",
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


SPECS = [
    ("M1", "configs/tuning/exp_m1_diffmul_supcon.yaml", "exp_m1_diffmul_supcon", "bcb-ablation-m1-diffmul-supcon", "BCB Ablation M1 DiffMul SupCon"),
    ("M2", "configs/tuning/exp_m2_diffmul_supcon.yaml", "exp_m2_diffmul_supcon", "bcb-ablation-m2-diffmul-supcon", "BCB Ablation M2 DiffMul SupCon"),
]


def main() -> None:
    payload = build_repo_payload()
    print(f"Payload: {len(payload)/1024:.1f} KB")
    for model, cfg, exp_id, slug, title in SPECS:
        nb = make_notebook(payload, model, cfg, exp_id)
        text = json.dumps(nb, indent=1, ensure_ascii=False)
        deploy = ROOT / "kaggle_deploy" / f"ablation_{model.lower()}_supcon"
        deploy.mkdir(parents=True, exist_ok=True)
        nb_name = f"bcb_ablation_{model.lower()}_supcon.ipynb"
        (deploy / nb_name).write_text(text, encoding="utf-8")
        (ROOT / "notebooks" / nb_name).write_text(text, encoding="utf-8")
        metadata = {
            "id": f"onngctrnchu/{slug}",
            "title": title,
            "code_file": nb_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": ["onngctrnchu/bcb-splits-30k-3k-5k"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (deploy / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Created {deploy / nb_name} | onngctrnchu/{slug}")


if __name__ == "__main__":
    main()
