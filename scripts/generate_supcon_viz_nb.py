"""Notebook Kaggle: trực quan hóa phân bố lớp có/không có SupCon."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "onngctrnchu/bcb-supcon-embedding-visualization"
KERNEL_TITLE = "BCB SupCon Embedding Visualization"
NOTEBOOK_NAME = "bcb_supcon_embedding_viz.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "supcon_viz"

# Dataset sẽ tạo khi shell ổn: chứa 2 checkpoint + bcb_test.parquet
DATASET_ID = "onngctrnchu/bcb-supcon-viz-ckpts"


def build_repo_payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("src", "scripts", "configs"):
            tar.add(ROOT / folder, arcname=folder, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_notebook(payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title",
            "metadata": {},
            "source": [
                "# Trực quan hóa phân bố lớp: có SupCon vs không có SupCon\n",
                "\n",
                "So sánh cosine giữa hai vector CLS của cặp clone/non-clone, AUC khi chỉ dùng cosine, t-SNE trên hA.\n",
                "Chỉ phân tích — không tune ngưỡng trên test.\n",
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
                "CLEAN = WORK / 'clean_viz'\n",
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
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow matplotlib')\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-link",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "data_raw = repo_root / 'data/raw/bigclonebench'\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "ckpts_dir = repo_root / 'results/ckpts'\n",
                "ckpts_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "files = {\n",
                "    'bcb_test.parquet': data_raw / 'bcb_test.parquet',\n",
                "    'm2_diffmul_best.pt': ckpts_dir / 'm2_diffmul_best.pt',\n",
                "    'm2_b5b6_best.pt': ckpts_dir / 'm2_b5b6_best.pt',\n",
                "}\n",
                "for name, dest in files.items():\n",
                "    cands = list(Path('/kaggle/input').rglob(name))\n",
                "    if cands and not dest.exists(): sh(f'cp \"{cands[0]}\" \"{dest}\"')\n",
                "    print(name, dest.exists(), flush=True)\n",
                "    assert dest.exists(), f'Thieu {name}'\n",
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
                "print('>>> BAT DAU SUP- CON EMBEDDING ANALYSIS <<<')\n",
                "print('=' * 70)\n",
                "t0 = time.perf_counter()\n",
                "sh('python scripts/analyze_supcon_embeddings.py '\n",
                "   '--no-supcon results/ckpts/m2_diffmul_best.pt '\n",
                "   '--with-supcon results/ckpts/m2_b5b6_best.pt '\n",
                "   '--pairs-file data/raw/bigclonebench/bcb_test.parquet '\n",
                "   '--out-dir results/supcon_embedding --batch-size 32')\n",
                "print(f'>>> XONG TRONG {(time.perf_counter()-t0)/60:.1f} PHUT', flush=True)\n",
                "print((repo_root / 'results/supcon_embedding/report.json').read_text(encoding='utf-8'), flush=True)\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-collect",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "out_dir = repo_root / 'results/supcon_embedding'\n",
                "for p in out_dir.rglob('*'):\n",
                "    if p.is_file() and p.suffix in ('.json', '.png', '.npy'):\n",
                "        name = 'viz_' + p.name\n",
                "        shutil.copy2(p, CLEAN / name)\n",
                "        shutil.copy2(p, WORK / name)\n",
                "        print('  [SAVED]', name, flush=True)\n",
                "log = WORK / 'live_progress.log'\n",
                "if log.exists():\n",
                "    shutil.copy2(log, CLEAN / 'viz_live_progress.log')\n",
                "    shutil.copy2(log, WORK / 'viz_live_progress.log')\n",
                "with tarfile.open(WORK / 'supcon_viz_bundle.tar.gz', 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'): tar.add(f, arcname=f.name)\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'data', 'clean_viz'):\n",
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
        "dataset_sources": [DATASET_ID, "onngctrnchu/bcb-splits-30k-3k-5k"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Created {DEPLOY_DIR / NOTEBOOK_NAME} | {KERNEL_ID}")


if __name__ == "__main__":
    main()
