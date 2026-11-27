"""Tạo notebook Kaggle chạy A1 -> A2 -> A3 cho M1/M2 DiffMul.

Checkpoint được lấy từ output kernel `bcb-tuning-m1-m2-diffmul-paired` qua
`kernel_sources` (không cần upload lại ~2GB). Parquet val/test lấy từ dataset
`kichirou/semanticcodeclonedetection-dataset-m4` qua `dataset_sources`.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent

KERNEL_ID = "tranthikhanhlinh1709/bcb-diffmul-postprocess-a1-a2-a3"
KERNEL_TITLE = "BCB DiffMul Postprocess A1 A2 A3"
NOTEBOOK_NAME = "bcb_diffmul_postprocess.ipynb"
DEPLOY_DIR = ROOT / "kaggle_deploy" / "postprocess_diffmul"


def build_repo_payload() -> str:
    """Nén src/, scripts/, configs/ thành base64 tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for folder in ("src", "scripts", "configs"):
            tar.add(
                ROOT / folder,
                arcname=folder,
                filter=lambda ti: None if "__pycache__" in ti.name or ti.isdir() and ti.name == "hf_cache" else ti,
            )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_notebook(payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-postprocess",
            "metadata": {},
            "source": [
                "# A1 → A2 → A3: hậu xử lý M1/M2 DiffMul trên Kaggle\n",
                "\n",
                "- **A1**: calibration (temperature / Platt / isotonic) + chọn ngưỡng 5-fold OOF trên validation.\n",
                "- **A2**: SWA-lite (trung bình best + last checkpoint), chọn biến thể bằng validation.\n",
                "- **A3**: ensemble M1 + M2, quét trọng số trên OOF validation.\n",
                "\n",
                "Checkpoint được gắn từ kernel `bcb-tuning-m1-m2-diffmul-paired`; parquet validation/test được gắn từ dataset.\n",
                "Mọi ngưỡng/trọng số chốt trên validation — test chỉ đánh giá cuối kèm bootstrap CI + McNemar.\n",
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
                "print('CPU count:', os.cpu_count(), flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack-repo",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIẢI NÉN MÃ NGUỒN\n",
                f"payload = '{payload_b64}'\n",
                "repo_root = Path('/kaggle/working/project')\n",
                "repo_root.mkdir(parents=True, exist_ok=True)\n",
                "with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode='r:gz') as tar:\n",
                "    tar.extractall(repo_root)\n",
                "os.chdir(repo_root)\n",
                "print('Repo root:', os.getcwd(), flush=True)\n",
                "sh('ls -la src/postprocess scripts/postprocess_diffmul.py')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-gpu-deps",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. GPU + DEPENDENCIES\n",
                "import torch\n",
                "print('CUDA:', torch.cuda.is_available(), flush=True)\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU:', torch.cuda.get_device_name(0), flush=True)\n",
                "    print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), flush=True)\n",
                "else:\n",
                "    raise SystemError('Kernel chưa bật GPU!')\n",
                "sh('pip install -q -U transformers datasets scikit-learn scipy pyarrow')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-link-data-ckpt",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT PARQUET VAL/TEST VÀ CHECKPOINT TỪ KERNEL SOURCE\n",
                "input_path = Path('/kaggle/input')\n",
                "print('Input dirs:', [p.name for p in input_path.iterdir()], flush=True)\n",
                "\n",
                "data_raw = Path('data/raw/bigclonebench')\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "for name in ('bcb_validation.parquet', 'bcb_test.parquet'):\n",
                "    candidates = list(input_path.rglob(name))\n",
                "    dest = data_raw / name\n",
                "    if candidates and not dest.exists():\n",
                "        sh(f'cp \"{candidates[0]}\" \"{dest}\"')\n",
                "    print(f'{name}: exists={dest.exists()}, size={dest.stat().st_size/1e6:.1f} MB' if dest.exists() else f'{name}: MISSING', flush=True)\n",
                "\n",
                "# Dataset Kaggle không chứa parquet -> tải lại đúng split đã dùng lúc train\n",
                "if any(not (data_raw / n).exists() for n in ('bcb_validation.parquet', 'bcb_test.parquet')):\n",
                "    print('Parquet chưa có trong input -> tải từ Hugging Face...', flush=True)\n",
                "    sh('python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000')\n",
                "\n",
                "assert (data_raw / 'bcb_validation.parquet').exists(), 'Thiếu bcb_validation.parquet!'\n",
                "assert (data_raw / 'bcb_test.parquet').exists(), 'Thiếu bcb_test.parquet!'\n",
                "\n",
                "# Copy checkpoint từ dataset chuyên dụng (đã upload)\n",
                "dataset_ckpts = {\n",
                "    'm1_best_checkpoint.pt': 'results/exp_m1_diffmul/best_checkpoint.pt',\n",
                "    'm2_best_checkpoint.pt': 'results/exp_m2_diffmul/best_checkpoint.pt',\n",
                "    'm2_last_checkpoint.pt': 'results/exp_m2_diffmul/last_checkpoint.pt',\n",
                "}\n",
                "for f in input_path.rglob('*.pt'):\n",
                "    if f.name in dataset_ckpts:\n",
                "        dest = dataset_ckpts[f.name]\n",
                "        Path(dest).parent.mkdir(parents=True, exist_ok=True)\n",
                "        if not Path(dest).exists():\n",
                "            shutil.copy2(f, dest)\n",
                "        print(f'Dataset ckpt: {f.name} -> {dest}', flush=True)\n",
                "\n",
                "# Fallback: bundle/output kernel cũ (nếu dataset chưa có)\n",
                "for tar_f in input_path.rglob('*.tar.gz'):\n",
                "    low = tar_f.name.lower()\n",
                "    if 'diff' in low or 'tuning' in low or 'bundle' in low:\n",
                "        print(f'Giải nén {tar_f} ...', flush=True)\n",
                "        sh(f'tar -xzf \"{tar_f}\" -C /kaggle/working/')\n",
                "for f in list(input_path.rglob('*.pt')) + list(Path('/kaggle/working').glob('exp_*_checkpoint.pt')):\n",
                "    low = str(f).lower()\n",
                "    dest = None\n",
                "    if 'exp_m1_diffmul_best' in low: dest = 'results/exp_m1_diffmul/best_checkpoint.pt'\n",
                "    elif 'exp_m2_diffmul_best' in low: dest = 'results/exp_m2_diffmul/best_checkpoint.pt'\n",
                "    elif 'exp_m2_diffmul_last' in low: dest = 'results/exp_m2_diffmul/last_checkpoint.pt'\n",
                "    if dest and not Path(dest).exists():\n",
                "        Path(dest).parent.mkdir(parents=True, exist_ok=True)\n",
                "        shutil.copy2(f, dest)\n",
                "\n",
                "# M1 last không cần: best epoch 8 == last epoch 8, pipeline tự bỏ qua SWA.\n",
                "required = list(dataset_ckpts.values())\n",
                "for dest in required:\n",
                "    p = Path(dest)\n",
                "    print(f'{dest}: exists={p.exists()}' + (f', {p.stat().st_size/1e6:.1f} MB' if p.exists() else ''), flush=True)\n",
                "    assert p.exists(), f'Thiếu checkpoint {dest}'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-run-postprocess",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. CHẠY A1 -> A2 -> A3\n",
                "print('=' * 70)\n",
                "print('>>> BAT DAU POSTPROCESS A1 -> A2 -> A3 <<<')\n",
                "print('=' * 70)\n",
                "cmd = (\n",
                "    'python scripts/postprocess_diffmul.py '\n",
                "    '--val-file data/raw/bigclonebench/bcb_validation.parquet '\n",
                "    '--test-file data/raw/bigclonebench/bcb_test.parquet '\n",
                "    '--out-dir results/exp_diffmul_postprocess '\n",
                "    '--batch-size 32 --precision fp16 --n-folds 5 --seed 42'\n",
                ")\n",
                "t0 = time.perf_counter()\n",
                "sh(cmd)\n",
                "print(f'>>> HOAN TAT TRONG {(time.perf_counter()-t0)/60:.1f} PHUT!', flush=True)\n",
                "print(Path('results/exp_diffmul_postprocess/postprocess_report.md').read_text(encoding='utf-8'), flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-collect-cleanup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. THU THẬP KẾT QUẢ + AUTO-CLEANUP\n",
                "WORK = Path('/kaggle/working')\n",
                "CLEAN = WORK / 'clean_postprocess'\n",
                "CLEAN.mkdir(parents=True, exist_ok=True)\n",
                "out_dir = repo_root / 'results' / 'exp_diffmul_postprocess'\n",
                "\n",
                "for p in sorted(out_dir.rglob('*')):\n",
                "    if p.is_file() and p.suffix in ('.json', '.md', '.npy', '.log'):\n",
                "        name = f'postprocess_{p.name}'\n",
                "        shutil.copy2(p, CLEAN / name)\n",
                "        shutil.copy2(p, WORK / name)\n",
                "        print(f'  [SAVED] {name}', flush=True)\n",
                "\n",
                "log_file = WORK / 'live_progress.log'\n",
                "if log_file.exists():\n",
                "    shutil.copy2(log_file, CLEAN / 'postprocess_live_progress.log')\n",
                "    shutil.copy2(log_file, WORK / 'postprocess_live_progress.log')\n",
                "\n",
                "bundle = WORK / 'diffmul_postprocess_bundle.tar.gz'\n",
                "with tarfile.open(bundle, 'w:gz') as tar:\n",
                "    for f in CLEAN.glob('*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "print(f'>>> Bundle: {bundle} ({bundle.stat().st_size/1e6:.1f} MB)', flush=True)\n",
                "\n",
                "os.chdir(WORK)\n",
                "for junk in ('project', 'data', 'artifacts', 'clean_postprocess', 'ckpt_dl'):\n",
                "    shutil.rmtree(WORK / junk, ignore_errors=True)\n",
                "# Xóa checkpoint/bundle tải về để không upload lại hàng GB khi kernel kết thúc\n",
                "for pattern in ('exp_m1_diffmul_*', 'exp_m2_diffmul_*', 'diff_mul_tuning_bundle.tar.gz'):\n",
                "    for p in WORK.glob(pattern):\n",
                "        if p.is_file():\n",
                "            p.unlink()\n",
                "print('DANH SÁCH OUTPUT:', flush=True)\n",
                "for p in sorted(WORK.glob('postprocess_*')):\n",
                "    print(f'  {p.name} ({p.stat().st_size/1e6:.2f} MB)', flush=True)\n",
                "print('>>> XONG 100%! <<<', flush=True)",
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
    nb_json = json.dumps(nb, indent=1, ensure_ascii=False)

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    (DEPLOY_DIR / NOTEBOOK_NAME).write_text(nb_json, encoding="utf-8")
    (ROOT / "notebooks" / NOTEBOOK_NAME).write_text(nb_json, encoding="utf-8")

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
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (DEPLOY_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Đã tạo: {DEPLOY_DIR / NOTEBOOK_NAME}")
    print(f"Kernel: {KERNEL_ID}")


if __name__ == "__main__":
    main()
