"""Notebook Kaggle: phân bố lớp DiffMul vs DiffMul+SupCon thuần cho M1/M2."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ID = "onngctrnchu/bcb-supcon-pure-visualization"
KERNEL_TITLE = "BCB SupCon Pure Visualization"
NOTEBOOK_NAME = "bcb_supcon_pure_viz.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "supcon_pure_viz"


def payload_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("src", "scripts", "configs"):
            tar.add(ROOT / folder, arcname=folder, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_nb(pl: str) -> dict:
    cells = [
        {"cell_type": "markdown", "id": "md", "metadata": {}, "source": [
            "# Phân bố lớp: DiffMul không SupCon vs DiffMul + SupCon thuần (M1/M2)\n",
            "Chỉ đổi đúng 1 yếu tố = SupCon. Ảnh 4 panel + bảng số liệu.\n"]},
        {"cell_type": "code", "id": "c1", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import base64, io, os, shutil, subprocess, sys, tarfile, time\n",
            "from pathlib import Path\n",
            "WORK = Path('/kaggle/working')\n",
            "CLEAN = WORK / 'clean_pure'\n", "CLEAN.mkdir(parents=True, exist_ok=True)\n",
            "def sh(cmd):\n",
            "    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
            "    with (WORK/'live_progress.log').open('a', encoding='utf-8') as lf:\n",
            "        for line in iter(p.stdout.readline, ''):\n",
            "            print(line, end='', flush=True); lf.write(line); lf.flush()\n",
            "    p.stdout.close()\n", "    rc = p.wait()\n",
            "    if rc != 0: raise RuntimeError(f'rc={rc}: {cmd}')\n"]},
        {"cell_type": "code", "id": "c2", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            f"payload = '{pl}'\n",
            "repo_root = WORK / 'project'\n", "repo_root.mkdir(parents=True, exist_ok=True)\n",
            "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as tar: tar.extractall(repo_root)\n",
            "os.chdir(repo_root)\n", "print(os.getcwd(), flush=True)\n"]},
        {"cell_type": "code", "id": "c3", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import torch\n", "print('CUDA', torch.cuda.is_available(), flush=True)\n",
            "if not torch.cuda.is_available(): raise SystemError('No GPU')\n",
            "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow matplotlib')\n"]},
        {"cell_type": "code", "id": "c4", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "data_raw = repo_root / 'data/raw/bigclonebench'\n", "data_raw.mkdir(parents=True, exist_ok=True)\n",
            "ckpt_dir = repo_root / 'results/ckpts'\n", "ckpt_dir.mkdir(parents=True, exist_ok=True)\n",
            "for f in Path('/kaggle/input').rglob('*.pt'):\n",
            "    dest = ckpt_dir / f.name\n",
            "    if not dest.exists(): shutil.copy2(f, dest)\n",
            "cands = list(Path('/kaggle/input').rglob('bcb_test.parquet'))\n",
            "if cands and not (data_raw/'bcb_test.parquet').exists(): shutil.copy2(cands[0], data_raw/'bcb_test.parquet')\n",
            "print('ckpts', len(list(ckpt_dir.glob('*.pt'))), 'test', (data_raw/'bcb_test.parquet').exists(), flush=True)\n"]},
        {"cell_type": "code", "id": "c5", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "t0 = time.perf_counter()\n",
            "sh('python scripts/analyze_supcon_pure.py --batch-size 32 --pairs-file data/raw/bigclonebench/bcb_test.parquet --out-dir results/supcon_pure')\n",
            "print(f'XONG {(time.perf_counter()-t0)/60:.1f} phut', flush=True)\n",
            "print((repo_root/'results/supcon_pure/report.json').read_text(encoding='utf-8'), flush=True)\n"]},
        {"cell_type": "code", "id": "c6", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "out = repo_root/'results/supcon_pure'\n",
            "for p in out.glob('*'):\n",
            "    if p.is_file() and p.suffix in ('.json','.png'):\n",
            "        name = 'pure_' + p.name\n", "        shutil.copy2(p, CLEAN/name); shutil.copy2(p, WORK/name)\n",
            "        print('SAVED', name, flush=True)\n",
            "os.chdir(WORK)\n",
            "for junk in ('project','data','clean_pure'): shutil.rmtree(WORK/junk, ignore_errors=True)\n",
            "print('DONE', flush=True)\n"]},
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}, "accelerator": "gpu"},
        "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    pl = payload_b64()
    print("Payload", len(pl) / 1024, "KB")
    nb = make_nb(pl)
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / NOTEBOOK_NAME).write_text(text, encoding="utf-8")
    (ROOT / "notebooks" / NOTEBOOK_NAME).write_text(text, encoding="utf-8")
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps({
        "id": KERNEL_ID, "title": KERNEL_TITLE, "code_file": NOTEBOOK_NAME,
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
        "dataset_sources": ["onngctrnchu/bcb-pure-ablation-ckpts"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2), encoding="utf-8")
    print("Created", KERNEL_ID)


if __name__ == "__main__":
    main()
