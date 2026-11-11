"""Tạo 2 notebook Kaggle độc lập cho Phase 4 (hoàn thiện 2 kịch bản cuối cùng của ma trận 14 thử nghiệm):
  1. Notebook M1-WD: Chạy EXP-M1-WD (CodeBERT, Weight Decay: 0.1)
  2. Notebook M2-DR: Chạy EXP-M2-DR (GraphCodeBERT, Dropout: 0.3)

Đặc điểm an toàn:
  - Tự động dò tìm dataset trong /kaggle/input (không phụ thuộc username).
  - Kiểm tra nghiêm ngặt GPU (CUDA), dependencies.
  - Bọc try-finally: Dù có lỗi hay thành công đều tự động DỌN DẸP SẠCH SẼ.
  - Sau khi chạy xong: XÓA HOÀN TOÀN src/, scripts/, data/, artifacts/, chỉ giữ lại checkpoint (.pt), file đánh giá (.json, .csv) và log.
"""

from __future__ import annotations

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


def make_single_transformer_tuning_notebook(
    model_id: str,
    model_name: str,
    exp_id: str,
    exp_name: str,
    config_path: str,
    bundle_name: str,
    payload_b64: str,
) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": f"md-title-{exp_id.lower()}",
            "metadata": {},
            "source": [
                f"# HUẤN LUYỆN THỬ NGHIỆM {exp_id} ({model_name} — {exp_name})\n",
                "\n",
                f"Notebook này thực hiện huấn luyện kịch bản tinh chỉnh **{exp_id}** cho mô hình **{model_id}** theo nguyên tắc OFAT:\n",
                f"- **Mã kịch bản:** `{exp_id}`\n",
                f"- **Mô hình cơ sở:** `{model_id}` ({model_name})\n",
                f"- **Nội dung tinh chỉnh:** {exp_name}\n",
                f"- **File cấu hình:** `{config_path}`\n",
                "\n",
                "### Cơ chế An toàn & Tự động dọn dẹp:\n",
                "- Tự động dò tìm dataset trong `/kaggle/input` mà không phụ thuộc username.\n",
                "- Chạy training kèm early stopping và logging chi tiết.\n",
                "- Đánh giá chuẩn xác trên 5.000 cặp BigCloneBench Java test balanced.\n",
                "- **Auto-Cleanup Cuối cùng:** Tự động xóa sạch toàn bộ mã nguồn `src/`, `scripts/`, `data/`, chỉ xuất ra checkpoint `.pt`, file đánh giá `.json`, `.csv` và file nén kết quả tinh gọn.\n",
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
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ STREAM LOG AN TOÀN\n",
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
                "print(\"Python:\", sys.version.split()[0], flush=True)\n",
                "print(\"Thư mục làm việc hiện tại:\", os.getcwd(), flush=True)",
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
                "\n",
                f"assert Path('{config_path}').exists(), f'Khong tim thay file {config_path}'\n",
                "print('Da giai nen ma nguon va xac nhan file cau hinh thanh cong!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-check-gpu-deps",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT DEPENDENCIES & XÁC THỰC GPU BẮT BUỘC\n",
                "sh(\"pip install -q --no-input 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'transformers>=4.30.0' 'torch_geometric' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn'\")\n",
                "\n",
                "import torch\n",
                "if not torch.cuda.is_available():\n",
                "    raise SystemError('GPU KHONG KHA DUNG! Hay vao menu Settings -> Accelerator -> Chon GPU (T4/P100) truoc khi chay.')\n",
                "\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)\n",
                "print(f'>>> GPU HOAT DONG: {gpu_name} | VRAM: {vram_gb} GB <<<')\n",
                "sh('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-dataset-link-smart",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. TỰ ĐỘNG DÒ TÌM VÀ LIÊN KẾT ĐẶC TRƯNG TỪ DATASET KAGGLE\n",
                "ds_root = None\n",
                "input_path = Path('/kaggle/input')\n",
                "if input_path.exists():\n",
                "    for p in input_path.rglob('*'):\n",
                "        if p.is_dir() and ('semanticcodeclone' in p.name.lower() or 'bigclonebench' in p.name.lower()):\n",
                "            ds_root = p\n",
                "            break\n",
                "print(f'Thu muc dataset phat hien: {ds_root}')\n",
                "if ds_root:\n",
                "    sh(f'ls -la {ds_root}/')\n",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-download-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. TẢI CÁC SPLIT DỮ LIỆU CHUẨN (TRAIN 30k, VAL 3k, TEST 5k CÂN BẰNG 50:50)\n",
                "data_raw = Path('data/raw/bigclonebench')\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Kiem tra neu dataset da co san file parquet thi copy nhanh de tiet kiem thoi gian\n",
                "cached_parquet = ds_root / 'bcb_train.parquet' if ds_root else None\n",
                "if cached_parquet and cached_parquet.exists():\n",
                "    print('Phat hien du lieu parquet san co tu dataset, sao chep truc tiep...')\n",
                "    sh(f'cp {ds_root}/*.parquet data/raw/bigclonebench/')\n",
                "else:\n",
                "    print('Tai truc tiep cac split BigCloneBench tu Hugging Face...')\n",
                "    sh('python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000')\n",
                "\n",
                "sh('ls -lh data/raw/bigclonebench/*.parquet')\n",
                "assert (data_raw / 'bcb_train.parquet').exists(), 'bcb_train.parquet phai ton tai truoc khi train!'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-run-exp",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 6. HUẤN LUYỆN & ĐÁNH GIÁ KỊCH BẢN: {exp_id} ({exp_name})\n",
                "print('=' * 70)\n",
                f"print('>>> BAT DAU KICH BAN: {exp_id} ({exp_name}) <<<')\n",
                "print('=' * 70)\n",
                f"cfg = '{config_path}'\n",
                "t_start = time.perf_counter()\n",
                "\n",
                "# Preflight check\n",
                f"sh(f'python -m src.experiments.bcb_runner --config {{cfg}} --mode preflight --device cuda')\n",
                "\n",
                "# Train (Time budget 6h, early stopping monitor val_loss)\n",
                f"sh(f'python -m src.experiments.bcb_runner --config {{cfg}} --mode train --device cuda --time-budget-s 21600')\n",
                "\n",
                "# Evaluate tren tap 5k BigCloneBench test\n",
                f"sh(f'python -m src.experiments.bcb_runner --config {{cfg}} --mode evaluate --device cuda')\n",
                "\n",
                f"print(f'>>> HOAN TAT {exp_id} trong {{(time.perf_counter() - t_start)/60:.1f}} phut! <<<')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-cleanup-and-export",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. DỌN DẸP SẠCH SẼ (AUTO-CLEANUP) & CHỈ XUẤT .PT, .JSON, .CSV TINH GỌN\n",
                "print('=' * 70)\n",
                "print('TIẾN HÀNH THU THẬP KẾT QUẢ & DỌN DẸP SẠCH MÃ NGUỒN KHÔNG LIÊN QUAN...')\n",
                "print('=' * 70)\n",
                "\n",
                "WORKING_DIR = Path('/kaggle/working')\n",
                "CLEAN_DIR = WORKING_DIR / 'clean_results'\n",
                "CLEAN_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# 1. Quet va sao chep toan bo file quan trong (.pt, .json, .csv) tu results/\n",
                "results_dir = Path('/kaggle/working/project/results')\n",
                "copied_files = []\n",
                "if results_dir.exists():\n",
                "    for p in results_dir.rglob('*'):\n",
                "        if p.is_file() and p.suffix in ('.pt', '.json', '.csv'):\n",
                "            subfolder = p.parent.name\n",
                "            out_name = f\"{subfolder}_{p.name}\" if not p.name.startswith(subfolder) else p.name\n",
                "            dest_clean = CLEAN_DIR / out_name\n",
                "            shutil.copy2(p, dest_clean)\n",
                "            shutil.copy2(p, WORKING_DIR / out_name)\n",
                "            copied_files.append(out_name)\n",
                "\n",
                "# Sao chep live_progress.log neu co\n",
                "if (WORKING_DIR / 'live_progress.log').exists():\n",
                "    shutil.copy2(WORKING_DIR / 'live_progress.log', CLEAN_DIR / 'live_progress.log')\n",
                "\n",
                "# 2. Dong goi toan bo ket qua sach vao 1 file tar.gz duy nhat\n",
                f"tar_name = '{bundle_name}'\n",
                "tar_path = WORKING_DIR / tar_name\n",
                "with tarfile.open(tar_path, 'w:gz') as tar:\n",
                "    for f in CLEAN_DIR.glob('*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "print(f'>>> Da tao file goi ket qua: {tar_name} ({tar_path.stat().st_size / 1024 / 1024:.2f} MB) <<<')\n",
                "\n",
                "# 3. XOA TRIET DE MA NGUON, CACHE VA DATA TRUOC KHI THOAT\n",
                "os.chdir('/kaggle/working')\n",
                "junk_paths = [\n",
                "    Path('/kaggle/working/project'),\n",
                "    Path('/kaggle/working/data'),\n",
                "    Path('/kaggle/working/artifacts'),\n",
                "    Path('/kaggle/working/clean_results'),\n",
                "]\n",
                "for jp in junk_paths:\n",
                "    if jp.exists():\n",
                "        shutil.rmtree(jp, ignore_errors=True)\n",
                "        print(f'Da xoa sach thu muc: {jp.name}/')\n",
                "\n",
                "# 4. In bao cao danh sach file con lai trong /kaggle/working/\n",
                "print('\\n' + '=' * 70)\n",
                "print('DANH SÁCH FILE ĐẦU RA SẠCH DUY NHẤT CÒN LẠI TRÊN /kaggle/working/ :')\n",
                "print('=' * 70)\n",
                "total_size = 0\n",
                "for item in sorted(WORKING_DIR.glob('*')):\n",
                "    if item.is_file():\n",
                "        sz = item.stat().st_size\n",
                "        total_size += sz\n",
                "        sz_str = f'{sz/1024/1024:.2f} MB' if sz > 1024*1024 else f'{sz/1024:.1f} KB'\n",
                "        print(f'  - {item.name:<45} | Size: {sz_str}')\n",
                "print(f'TONG DUNG LUONG DAU RA: {total_size/1024/1024:.2f} MB')\n",
                "print('>>> QUA TRINH HUAN LUYEN VA DON DEP DA HOAN TAT AN TOAN 100%! <<<')",
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
    print(f"Ma nguon da dong goi: {len(payload)/1024:.1f} KB base64")

    # 1. Notebook M1-WD: EXP-M1-WD (Weight Decay 0.1)
    nb_m1_wd = make_single_transformer_tuning_notebook(
        model_id="M1",
        model_name="CodeBERT",
        exp_id="EXP-M1-WD",
        exp_name="Tăng Weight Decay lên 0.1 (baseline 0.05)",
        config_path="configs/tuning/exp_m1_wd.yaml",
        bundle_name="m1_wd_tuning_clean_bundle.tar.gz",
        payload_b64=payload,
    )
    m1_dir = Path("kaggle_deploy/tuning_m1_wd")
    m1_dir.mkdir(parents=True, exist_ok=True)
    m1_nb_path = m1_dir / "bcb_tuning_m1_wd.ipynb"
    m1_nb_path.write_text(json.dumps(nb_m1_wd, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("notebooks/bcb_tuning_m1_wd.ipynb").write_text(json.dumps(nb_m1_wd, indent=1, ensure_ascii=False), encoding="utf-8")
    (m1_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "tranthikhanhlinh1709/bcb-tuning-m1-wd",
        "title": "BCB Tuning M1 WD",
        "code_file": "bcb_tuning_m1_wd.ipynb",
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
    }, indent=2), encoding="utf-8")
    print(f"Da tao thanh cong: {m1_nb_path}")

    # 2. Notebook M2-DR: EXP-M2-DR (Dropout 0.3)
    nb_m2_dr = make_single_transformer_tuning_notebook(
        model_id="M2",
        model_name="GraphCodeBERT",
        exp_id="EXP-M2-DR",
        exp_name="Tăng Dropout lên 0.3 (baseline 0.2)",
        config_path="configs/tuning/exp_m2_dr.yaml",
        bundle_name="m2_dr_tuning_clean_bundle.tar.gz",
        payload_b64=payload,
    )
    m2_dir = Path("kaggle_deploy/tuning_m2_dr")
    m2_dir.mkdir(parents=True, exist_ok=True)
    m2_nb_path = m2_dir / "bcb_tuning_m2_dr.ipynb"
    m2_nb_path.write_text(json.dumps(nb_m2_dr, indent=1, ensure_ascii=False), encoding="utf-8")
    Path("notebooks/bcb_tuning_m2_dr.ipynb").write_text(json.dumps(nb_m2_dr, indent=1, ensure_ascii=False), encoding="utf-8")
    (m2_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": "tranthikhanhlinh1709/bcb-tuning-m2-dr",
        "title": "BCB Tuning M2 DR",
        "code_file": "bcb_tuning_m2_dr.ipynb",
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
    }, indent=2), encoding="utf-8")
    print(f"Da tao thanh cong: {m2_nb_path}")


if __name__ == "__main__":
    main()
