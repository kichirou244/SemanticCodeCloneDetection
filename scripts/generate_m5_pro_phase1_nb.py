"""Tạo Notebook Kaggle tự động cho Mô hình Lai Ghép M5-Pro Giai đoạn 1 (Phase 1).

Đóng gói toàn bộ mã nguồn vào payload base64 và sinh metadata sẵn sàng đẩy lên Kaggle.
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


def make_m5_pro_phase1_notebook(payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-m5-pro-phase1",
            "metadata": {},
            "source": [
                "# HUẤN LUYỆN M5-PRO GIAI ĐOẠN 1: TỐI ƯU HÀM MẤT MÁT TRÊN 30.000 CẶP DỮ LIỆU\n",
                "\n",
                "Notebook này thực hiện huấn luyện và so sánh đối chứng **4 biến thể mất mát tiên tiến** nhằm triệt tiêu báo nhầm False Positives của M5-Gated mà không làm suy giảm kỷ lục Recall 96.72%:\n",
                "1. **M5_Gated_BCE:** BCEWithLogitsLoss (Baseline hiện tại).\n",
                "2. **M5_Gated_Focal_gamma2:** Focal Loss (gamma=2.0, alpha=0.25).\n",
                "3. **M5_Gated_Focal_gamma3:** Focal Loss (gamma=3.0, alpha=0.25).\n",
                "4. **M5_Gated_Asymmetric_Loss:** Asymmetric Loss (gamma_neg=4.0, gamma_pos=1.0) phạt bất đối xứng vào False Positives.\n",
                "\n",
                "Đánh giá độc lập trên cùng 5.000 cặp BigCloneBench Test độc lập chuẩn 1:1.\n",
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
                "assert Path('src/experiments/train_m5_pro_phase1.py').exists(), 'Thiếu file train_m5_pro_phase1.py'\n",
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
            "id": "c4-link-cpg-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. TỰ ĐỘNG DÒ TÌM & LIÊN KẾT 9.122 ĐỒ THỊ CPG TỪ DATASET KAGGLE\n",
                "input_path = Path('/kaggle/input')\n",
                "os.makedirs('artifacts/cpg_export', exist_ok=True)\n",
                "cpg_found = 0\n",
                "\n",
                "for cand in input_path.rglob('cpg_export'):\n",
                "    if cand.is_dir() and any(cand.glob('*.json')):\n",
                "        dest = Path('artifacts/cpg_export')\n",
                "        print(f'Phát hiện thư mục CPG: {cand}. Tiến hành sao chép...')\n",
                "        sh(f'cp -r {cand}/* artifacts/cpg_export/')\n",
                "        cpg_found = len(list(dest.glob('*.json')))\n",
                "        print(f'Đã liên kết {cpg_found:,} đồ thị CPG vào artifacts/cpg_export!')\n",
                "        break\n",
                "\n",
                "if cpg_found == 0:\n",
                "    for tar_f in input_path.rglob('*.tar.gz'):\n",
                "        if 'cpg' in tar_f.name.lower():\n",
                "            sh(f'tar -xzf {tar_f} -C artifacts/')\n",
                "            cpg_found = len(list(Path('artifacts/cpg_export').glob('*.json')))\n",
                "            print(f'Đã giải nén {cpg_found:,} đồ thị CPG từ {tar_f.name}!')\n",
                "            break\n",
                "\n",
                "assert cpg_found > 0, 'Lỗi nghiêm trọng: Không tìm thấy tập đồ thị CPG trong /kaggle/input!'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-link-warmstart",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. DÒ TÌM VÀ LIÊN KẾT CHECKPOINT KHỞI TẠO ẤM TỪ EXP-M2-DR VÀ EXP-M4-EP\n",
                "os.makedirs('results/exp_m2_dr', exist_ok=True)\n",
                "os.makedirs('results/exp_m4_ep', exist_ok=True)\n",
                "\n",
                "m2_src, m4_src = None, None\n",
                "for f in input_path.rglob('*.pt'):\n",
                "    if 'm2_dr' in f.name.lower() and 'best' in f.name.lower():\n",
                "        m2_src = f\n",
                "    elif 'm4_ep' in f.name.lower() and 'best' in f.name.lower():\n",
                "        m4_src = f\n",
                "\n",
                "if m2_src and m2_src.exists():\n",
                "    sh(f'cp {m2_src} results/exp_m2_dr/best_checkpoint.pt')\n",
                "    print(f'Đã nạp checkpoint ấm M2-DR: {m2_src.name} ({m2_src.stat().st_size/1e6:.1f} MB)')\n",
                "else:\n",
                "    print('Cảnh báo: Không tìm thấy checkpoint M2-DR trong input!')\n",
                "\n",
                "if m4_src and m4_src.exists():\n",
                "    sh(f'cp {m4_src} results/exp_m4_ep/best_checkpoint.pt')\n",
                "    print(f'Đã nạp checkpoint ấm M4-EP: {m4_src.name} ({m4_src.stat().st_size/1e6:.1f} MB)')\n",
                "else:\n",
                "    print('Cảnh báo: Không tìm thấy checkpoint M4-EP trong input!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-prepare-dataset-splits",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. CHUẨN BỊ DỮ LIỆU ĐỒNG NHẤT (TRAIN 30K, VAL 3K, TEST 5K)\n",
                "print('Bắt đầu chuẩn bị dữ liệu chuẩn hóa 30k train, 3k val, 5k test...')\n",
                "sh('python scripts/prepare_hybrid_level3_data.py')\n",
                "\n",
                "assert Path('data/raw/bigclonebench/train_shard_0.parquet').exists(), 'Thiếu train_shard_0.parquet'\n",
                "assert Path('data/raw/bigclonebench/validation_shard_0.parquet').exists(), 'Thiếu validation_shard_0.parquet'\n",
                "assert Path('results/bcb_5k_test_pairs.jsonl').exists(), 'Thiếu bcb_5k_test_pairs.jsonl'\n",
                "print('Xác nhận toàn bộ các file nguồn dữ liệu đã sẵn sàng!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-run-phase1-training",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. THỰC THI HUẤN LUYỆN 4 BIẾN THỂ MẤT MÁT TRÊN 30.000 CẶP\n",
                "print('=' * 80)\n",
                "print('>>> BẮT ĐẦU CHẠY PIPELINE M5-PRO PHASE 1 TRÊN 30.000 CẶP <<<')\n",
                "print('=' * 80)\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "sh('python -m src.experiments.train_m5_pro_phase1 --output-dir results/exp_m5_pro_phase1 --epochs 15 --batch_size 64')\n",
                "\n",
                "t_dur = (time.perf_counter() - t0) / 60\n",
                "print(f'>>> HOÀN TẤT PIPELINE M5-PRO PHASE 1 TRONG {t_dur:.1f} PHÚT! <<<')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c8-cleanup-and-export",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. THU THẬP KẾT QUẢ VÀ DỌN DẸP SẠCH KHÔNG GIAN ĐĨA\n",
                "print('=' * 80)\n",
                "print('THU THẬP KẾT QUẢ & ĐÓNG GÓI BUNDLE TINH GỌN...')\n",
                "print('=' * 80)\n",
                "\n",
                "WORKING_DIR = Path('/kaggle/working')\n",
                "exp_dir = Path('/kaggle/working/project/results/exp_m5_pro_phase1')\n",
                "\n",
                "for f in exp_dir.glob('*'):\n",
                "    if f.is_file():\n",
                "        dest_f = WORKING_DIR / f'exp_m5_pro_{f.name}'\n",
                "        shutil.copy2(f, dest_f)\n",
                "        print(f'  -> Xuất file: {dest_f.name} ({dest_f.stat().st_size/1e6:.2f} MB)')\n",
                "\n",
                "# Dong goi bundle kết quả\n",
                "tar_name = 'bcb_m5_pro_phase1_bundle.tar.gz'\n",
                "tar_path = WORKING_DIR / tar_name\n",
                "with tarfile.open(tar_path, 'w:gz') as tar:\n",
                "    for f in WORKING_DIR.glob('exp_m5_pro_*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "    if (WORKING_DIR / 'live_progress.log').exists():\n",
                "        tar.add(WORKING_DIR / 'live_progress.log', arcname='live_progress.log')\n",
                "\n",
                "print(f'Đã đóng gói bundle hoàn tất: {tar_path} ({tar_path.stat().st_size/1e6:.2f} MB)')\n",
                "# Dọn dẹp thư mục tạm để giải phóng đĩa\n",
                "shutil.rmtree(WORK_DIR, ignore_errors=True)\n",
                "print('Đã dọn dẹp thư mục tạm thành công!')",
            ],
        },
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "gpuClass": "standard",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    deploy_dir = Path("kaggle_deploy/m5_pro_phase1")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    print("Đang nén mã nguồn dự án thành base64 payload...")
    payload_b64 = build_repo_payload()
    print(f"Kích thước payload: {len(payload_b64) / 1024:.2f} KB")

    print("Đang sinh notebook Kaggle...")
    nb_dict = make_m5_pro_phase1_notebook(payload_b64)
    nb_path = deploy_dir / "bcb_m5_pro_phase1_kaggle.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu notebook: {nb_path} ({nb_path.stat().st_size / 1024:.2f} KB)")

    meta = {
        "id": "tranthikhanhlinh1709/bcb-m5-pro-phase-1-focal-asl-30k",
        "title": "BCB M5 Pro Phase 1 Focal ASL 30k",
        "code_file": "bcb_m5_pro_phase1_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "kichirou/semanticcodeclonedetection-dataset-m4"
        ],
        "kernel_sources": [
            "tranthikhanhlinh1709/bcb-tuning-m2-dr",
            "tranthikhanhlinh1709/bcb-tuning-m4-cpg-gnn"
        ],
        "competition_sources": [],
        "model_sources": [],
    }
    meta_path = deploy_dir / "kernel-metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu metadata: {meta_path}")
    print("\n>>> SẴN SÀNG ĐẨY LÊN KAGGLE! <<<")


if __name__ == "__main__":
    main()
