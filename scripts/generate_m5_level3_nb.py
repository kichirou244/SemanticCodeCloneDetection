"""Tạo Notebook Kaggle tự động cho Mô hình Lai Ghép Cấp độ 3 (Deep Graph-Transformer M5).

Kiến trúc liên kết ngữ nghĩa sâu End-to-End giữa GraphCodeBERT và CPG-GNN.
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


def make_hybrid_level3_notebook(payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title-hybrid-level3",
            "metadata": {},
            "source": [
                "# HUẤN LUYỆN MÔ HÌNH LAI GHÉP CẤP ĐỘ 3: DEEP GRAPH-TRANSFORMER (M5)\n",
                "\n",
                "Notebook này thực hiện huấn luyện mô hình **M5 Cấp độ 3 (End-to-End Deep Graph-Transformer)**:\n",
                "- **Mô hình Văn bản:** `GraphCodeBERT` (125M tham số) trích xuất đặc trưng ngữ nghĩa chuỗi mã nguồn.\n",
                "- **Mô hình Đồ thị:** `CPG-GNN` (3 lớp Typed GAT với nhúng quan hệ 32 chiều) trích xuất cấu trúc AST + CFG + CDG + DDG.\n",
                "- **Cơ chế Hợp nhất:** `Bidirectional Dynamic Gated Cross-Modal Fusion` (Điều biến ngữ cảnh 2 chiều).\n",
                "- **Cơ chế Huấn luyện:** Joint Fine-tuning đồng thời cả 2 backbone với **Discriminative Learning Rates**:\n",
                "  * Transformer LR: `1.5e-5` (bảo toàn tri thức tiền huấn luyện)\n",
                "  * GNN LR: `1.0e-4` (thích ứng không gian đồ thị theo ngữ cảnh)\n",
                "  * Fusion & Head LR: `2.0e-4` (tối ưu hóa phân loại Siamese clone)\n",
                "- **Khởi tạo ấm (Warm-start):** Tự động liên kết trọng số tối ưu từ `EXP-M2-DR` và `EXP-M4-EP`.\n",
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
                "assert Path('configs/tuning/exp_m5_level3.yaml').exists(), 'Khong tim thay exp_m5_level3.yaml'\n",
                "print('Đã giải nén mã nguồn và xác nhận file cấu hình Level 3 thành công!')",
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
            "id": "c4-dataset-link-cpg",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. TỰ ĐỘNG DÒ TÌM & LIÊN KẾT 9.122 ĐỒ THỊ CPG TỪ DATASET KAGGLE\n",
                "input_path = Path('/kaggle/input')\n",
                "ds_root = None\n",
                "if input_path.exists():\n",
                "    for p in input_path.rglob('*'):\n",
                "        if p.is_dir() and ('semanticcodeclone' in p.name.lower() or 'cpg_export' in p.name.lower()):\n",
                "            ds_root = p\n",
                "            break\n",
                "\n",
                "os.makedirs('artifacts/cpg_export', exist_ok=True)\n",
                "cpg_found = 0\n",
                "\n",
                "# Tim thu muc bcb_cpg_export/cpg_export\n",
                "for cand in input_path.rglob('cpg_export'):\n",
                "    if cand.is_dir() and any(cand.glob('*.json')):\n",
                "        dest = Path('artifacts/cpg_export')\n",
                "        print(f'Phat hien thu muc CPG: {cand}. Tien hanh sao chep...')\n",
                "        sh(f'cp -r {cand}/* artifacts/cpg_export/')\n",
                "        cpg_found = len(list(dest.glob('*.json')))\n",
                "        print(f'Da lien ket {cpg_found:,} do thi CPG vao artifacts/cpg_export!')\n",
                "        break\n",
                "\n",
                "if cpg_found == 0:\n",
                "    for tar_f in input_path.rglob('*.tar.gz'):\n",
                "        if 'cpg' in tar_f.name.lower():\n",
                "            sh(f'tar -xzf {tar_f} -C artifacts/')\n",
                "            cpg_found = len(list(Path('artifacts/cpg_export').glob('*.json')))\n",
                "            print(f'Da giai nen {cpg_found:,} do thi CPG tu {tar_f.name}!')\n",
                "            break\n",
                "\n",
                "assert cpg_found > 0, 'Loi nghiem trong: Khong tim thay tap do thi CPG trong /kaggle/input!'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-link-warmstart",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. DÒ TÌM VÀ LIÊN KẾT CHECKPOINT KHỞI TẠO ẤM (WARM-START) TỪ EXP-M2-DR VÀ EXP-M4-EP\n",
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
                "    print('Không tìm thấy checkpoint M2-DR trong input, sẽ khởi tạo trực tiếp từ microsoft/graphcodebert-base.')\n",
                "\n",
                "if m4_src and m4_src.exists():\n",
                "    sh(f'cp {m4_src} results/exp_m4_ep/best_checkpoint.pt')\n",
                "    print(f'Đã nạp checkpoint ấm M4-EP: {m4_src.name} ({m4_src.stat().st_size/1e6:.1f} MB)')\n",
                "else:\n",
                "    print('Không tìm thấy checkpoint M4-EP trong input, sẽ khởi tạo mới encoder CPG-GNN.')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-prepare-dataset-splits",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. CHUẨN BỊ & XÁC THỰC CÁC TẬP DỮ LIỆU CHUẨN (TRAIN 6K, VAL 2K, TEST 5K)\n",
                "print('Bắt đầu tải các shard Parquet và tạo 3 tập dữ liệu chuẩn...')\n",
                "sh('python scripts/prepare_hybrid_level3_data.py')\n",
                "\n",
                "assert Path('results/bcb_hybrid_train_6k.jsonl').exists(), 'Thieu train_6k.jsonl'\n",
                "assert Path('results/bcb_hybrid_val_2k.jsonl').exists(), 'Thieu val_2k.jsonl'\n",
                "assert Path('results/bcb_5k_test_pairs.jsonl').exists(), 'Thieu test_5k.jsonl'\n",
                "print('Xác nhận 3 tập dữ liệu chuẩn đã sẵn sàng!')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-run-level3-training",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. HUẤN LUYỆN JOINT FINE-TUNING & ĐÁNH GIÁ KIỂM ĐỊNH MÔ HÌNH CẤP ĐỘ 3\n",
                "print('=' * 75)\n",
                "print('>>> BẮT ĐẦU HUẤN LUYỆN JOINT FINE-TUNING M5 LEVEL 3 <<<')\n",
                "print('=' * 75)\n",
                "\n",
                "t0 = time.perf_counter()\n",
                "sh('python -m src.experiments.train_hybrid_level3 --epochs 6 --batch-size 16 --grad-accum 2 --lr-text 1.5e-5 --lr-graph 1e-4 --lr-head 2e-4 --output-dir results/exp_m5_level3')\n",
                "\n",
                "t_dur = (time.perf_counter() - t0) / 60\n",
                "print(f'>>> HOÀN TẤT HUẤN LUYỆN VÀ ĐÁNH GIÁ M5 LEVEL 3 TRONG {t_dur:.1f} PHÚT! <<<')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c8-cleanup-and-export",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 8. THU THẬP KẾT QUẢ & DỌN DẸP SẠCH SẼ (AUTO-CLEANUP)\n",
                "print('=' * 75)\n",
                "print('THU THẬP KẾT QUẢ VÀ DỌN DẸP SẠCH KHÔNG GIAN ĐĨA...')\n",
                "print('=' * 75)\n",
                "\n",
                "WORKING_DIR = Path('/kaggle/working')\n",
                "exp_dir = Path('/kaggle/working/project/results/exp_m5_level3')\n",
                "\n",
                "export_files = [\n",
                "    'best_checkpoint.pt',\n",
                "    'last_checkpoint.pt',\n",
                "    'test_metrics.json',\n",
                "    'training_summary.json',\n",
                "    'm5_level3_5k_probs.npy',\n",
                "]\n",
                "\n",
                "for fname in export_files:\n",
                "    src_f = exp_dir / fname\n",
                "    if src_f.exists():\n",
                "        dest_f = WORKING_DIR / f'exp_m5_level3_{fname}'\n",
                "        shutil.copy2(src_f, dest_f)\n",
                "        print(f'  -> Xuất file: {dest_f.name} ({dest_f.stat().st_size/1e6:.2f} MB)')\n",
                "\n",
                "# Dong goi ket qua vao tar.gz\n",
                "tar_name = 'bcb_hybrid_level3_bundle.tar.gz'\n",
                "tar_path = WORKING_DIR / tar_name\n",
                "with tarfile.open(tar_path, 'w:gz') as tar:\n",
                "    for f in WORKING_DIR.glob('exp_m5_level3_*'):\n",
                "        tar.add(f, arcname=f.name)\n",
                "    if (WORKING_DIR / 'live_progress.log').exists():\n",
                "        tar.add(WORKING_DIR / 'live_progress.log', arcname='live_progress.log')\n",
                "\n",
                "print(f'Đã đóng gói bundle kết quả: {tar_name} ({tar_path.stat().st_size/1e6:.2f} MB)')\n",
                "\n",
                "# Xoa du lieu trung gian de tiet kiem bo nho Kaggle\n",
                "os.chdir('/kaggle/working')\n",
                "for rm_dir in ['project/data', 'project/artifacts', 'project/results']:\n",
                "    p = Path(rm_dir)\n",
                "    if p.exists():\n",
                "        shutil.rmtree(p, ignore_errors=True)\n",
                "\n",
                "print('>>> ĐÃ HOÀN TẤT VÀ DỌN DẸP SẠCH SẼ! <<<')",
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
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    deploy_dir = Path("kaggle_deploy/hybrid_m5")
    deploy_dir.mkdir(parents=True, exist_ok=True)

    print("Đang nén mã nguồn dự án thành base64 payload...")
    payload_b64 = build_repo_payload()
    print(f"Kích thước payload: {len(payload_b64) / 1024:.2f} KB")

    print("Đang sinh notebook Kaggle...")
    nb_dict = make_hybrid_level3_notebook(payload_b64)
    nb_path = deploy_dir / "bcb_hybrid_level3_kaggle.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu notebook: {nb_path} ({nb_path.stat().st_size / 1024:.2f} KB)")

    # Sinh metadata
    meta = {
        "id": "tranthikhanhlinh1709/bcb-hybrid-level3-deep-graph-transformer",
        "title": "BCB Hybrid Level3 Deep Graph-Transformer",
        "code_file": "bcb_hybrid_level3_kaggle.ipynb",
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
