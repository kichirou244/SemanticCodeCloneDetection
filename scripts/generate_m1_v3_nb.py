"""Tạo notebook huấn luyện cho M1 (CodeBERT v3) và M2 (GraphCodeBERT v3) trên Kaggle GPU kèm Regularization & Label Smoothing."""

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


def make_v3_notebook(model_id: str, model_name: str, config_file: str, payload_b64: str) -> dict:
    cells = [
        {
            "cell_type": "markdown",
            "id": f"md-title-{model_id.lower()}-v3",
            "metadata": {},
            "source": [
                f"# HUẤN LUYỆN {model_id} ({model_name}) MAX_LEN 512 V3 (REGULARIZED + LABEL SMOOTHING)\n",
                "\n",
                f"Notebook này thực hiện huấn luyện mô hình **{model_id} ({model_name}) phiên bản V3** trên BigCloneBench Java với các cải tiến chống overfit mạnh mẽ:\n",
                "- **Tokenizer max_length:** 512 (dynamic padding)\n",
                "- **Label Smoothing:** 0.05 (giảm over-confidence của BCE loss)\n",
                "- **Dropout:** 0.2 (tăng tính tổng quát hóa)\n",
                "- **Weight Decay:** 0.05 (phạt trọng số lớn)\n",
                "- **Số epochs tối đa:** 15 epochs\n",
                "- **Early Stopping:** monitor `val_loss`, patience=5, min_delta=0.0005\n",
                "- **Dataset liên kết:** `semanticcodeclonedetection-dataset-m4`\n",
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
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ STREAMING LOG\n",
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
                "print(\"Python:\", sys.version.split()[0], flush=True)",
            ],
        },
        {
            "cell_type": "code",
            "id": "c2-unpack",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 2. GIẢI NÉN MÃ NGUỒN DỰ ÁN MỚI NHẤT ({model_id} V3)\n",
                "WORK_DIR = Path(\"/kaggle/working/project\")\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                f"CONFIG_FILE = \"{config_file}\"\n",
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
                "sh(f\"cat {CONFIG_FILE}\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c3-deps-gpu",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT THƯ VIỆN & XÁC THỰC GPU\n",
                "sh(\"pip install -q 'tree-sitter>=0.23.0' 'tree-sitter-java>=0.23.5' 'tree-sitter-cpp>=0.23.0' 'datasets>=2.14.0' 'transformers>=4.30.0' 'pyarrow' 'tqdm' 'pyyaml' 'scikit-learn' 'matplotlib'\")\n",
                "\n",
                "import torch\n",
                "assert torch.cuda.is_available(), \"Cần bật GPU trong Settings -> Accelerator -> GPU (T4/P100)!\"\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)\n",
                "print(f\"GPU SẴN SÀNG: {gpu_name} | VRAM: {vram_gb} GB\")\n",
                "sh(\"nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c4-dataset-link",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. LIÊN KẾT ĐẶC TRƯNG TỪ DATASET NẾU CÓ\n",
                "in_dir = Path(\"/kaggle/input/semanticcodeclonedetection-dataset-m4\")\n",
                "if in_dir.exists():\n",
                "    print(f\"Dataset đính kèm: {in_dir}\")\n",
                "    sh(f\"ls -lh {in_dir}\")\n",
                "else:\n",
                "    print(\"Chưa phát hiện dataset đính kèm, sử dụng dữ liệu trực tiếp.\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c5-download-data",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. TẢI CÁC SPLIT DỮ LIỆU ĐỂ HUẤN LUYỆN (TRAIN 30.000, VAL 3.000, TEST 5.000 CÂN BẰNG 50:50)\n",
                "data_raw = Path(\"data/raw/bigclonebench\")\n",
                "data_raw.mkdir(parents=True, exist_ok=True)\n",
                "print(\"Tải các phân tập dữ liệu BigCloneBench (Train: 30k, Val: 3k, Test: 5k) cân bằng 50:50...\")\n",
                "sh(\"python scripts/download_bcb_subsets.py --splits-custom --train-samples 30000 --val-samples 3000 --test-samples 5000\")\n",
                "sh(\"ls -lh data/raw/bigclonebench/\")\n",
                "assert (data_raw / 'bcb_train.parquet').exists() and (data_raw / 'bcb_train.parquet').stat().st_size > 0, 'bcb_train.parquet phai ton tai va co du lieu!'",
            ],
        },
        {
            "cell_type": "code",
            "id": "c6-preflight-diagnostic",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. PREFLIGHT & DIAGNOSTIC CHECK\n",
                f"print(\"--- PREFLIGHT {model_id} V3 ---\")\n",
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode preflight --device cuda\")\n",
                f"print(\"\\n--- DIAGNOSTIC CHECK {model_id} V3 ---\")\n",
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode diagnostic --device cuda\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c7-train",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 7. TIẾN HÀNH HUẤN LUYỆN {model_id} ({model_name}) V3 TRÊN GPU\n",
                "print(\"=\" * 60)\n",
                f"print(\"BẮT ĐẦU HUẤN LUYỆN {model_id} ({model_name}) MAX_LEN 512 V3...\")\n",
                "print(\"=\" * 60)\n",
                "TIME_BUDGET_S = 11 * 3600\n",
                f"try:\n",
                f"    sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode train --device cuda --time-budget-s {{TIME_BUDGET_S}}\")\n",
                f"finally:\n",
                f"    # Lưu ngay kết quả huấn luyện\n",
                f"    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:\n",
                f"        exp_id = yaml.safe_load(f)['experiment']['id'].lower().replace('-', '_')\n",
                f"    res_dir = Path('results') / exp_id\n",
                f"    dest_dir = Path('/kaggle/working/{model_id.lower()}_results')\n",
                f"    dest_dir.mkdir(parents=True, exist_ok=True)\n",
                f"    if res_dir.exists():\n",
                f"        for f in res_dir.glob('*'):\n",
                f"            sh(f'cp -r {{f}} {{dest_dir}}/')\n",
                f"            if f.suffix in ('.json', '.pt'):\n",
                f"                sh(f'cp {{f}} /kaggle/working/')\n",
                f"    sh(f'tar -czf /kaggle/working/{model_id.lower()}_results.tar.gz results/')\n",
                f"    print(f'>>> ĐÃ SAO LƯU CHECKPOINT VÀ KẾT QUẢ HUẤN LUYỆN CỦA {model_id} VÀO /kaggle/working/')",
            ],
        },
        {
            "cell_type": "code",
            "id": "c8-evaluate",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 8. ĐÁNH GIÁ MÔ HÌNH {model_id} V3 TRÊN TẬP TEST 5.000 CẶP\n",
                "print(\"=\" * 60)\n",
                f"print(\"ĐÁNH GIÁ MÔ HÌNH {model_id} V3 TRÊN TẬP TEST...\")\n",
                "print(\"=\" * 60)\n",
                f"sh(f\"python -m src.experiments.bcb_runner --config {{CONFIG_FILE}} --mode evaluate --device cuda\")",
            ],
        },
        {
            "cell_type": "code",
            "id": "c9-package-and-plot",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# 9. ĐÓNG GÓI, VẼ ĐƯỜNG LOSS & TỔNG KẾT KẾT QUẢ VÀO /kaggle/working/\n",
                "with open(CONFIG_FILE, 'r', encoding='utf-8') as f:\n",
                "    exp_id = yaml.safe_load(f)['experiment']['id'].lower().replace('-', '_')\n",
                "res_dir = Path('results') / exp_id\n",
                f"dest_dir = Path('/kaggle/working/{model_id.lower()}_results')\n",
                "dest_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "summary_file = res_dir / 'training_summary.json'\n",
                "if summary_file.exists():\n",
                f"    sh(f'python scripts/plot_loss_curves_30epochs.py --{model_id.lower()}-summary {{summary_file}} --out /kaggle/working/{model_id.lower()}_loss_curve.png')\n",
                "\n",
                "if res_dir.exists():\n",
                "    for f in res_dir.glob('*'):\n",
                "        sh(f'cp -r {f} {dest_dir}/')\n",
                "        if f.suffix in ('.json', '.pt', '.png'):\n",
                "            sh(f'cp {f} /kaggle/working/')\n",
                "\n",
                f"TAR_RESULTS = Path('/kaggle/working/{model_id.lower()}_results.tar.gz')\n",
                "sh(f'tar -czf {TAR_RESULTS} results/')\n",
                "sh('ls -lh /kaggle/working/*.json /kaggle/working/*.tar.gz /kaggle/working/*.png 2>/dev/null || ls -lh /kaggle/working/')\n",
                "\n",
                "if (res_dir / 'test_metrics.json').exists():\n",
                "    metrics = json.loads((res_dir / 'test_metrics.json').read_text(encoding='utf-8'))\n",
                f"    print('\\nBẢNG TEST METRICS CỦA {model_id} (max_len=512 V3):\\n', json.dumps(metrics, indent=2))\n",
                f"print('\\nHUẤN LUYỆN VÀ ĐÁNH GIÁ {model_id} V3 HOÀN TẤT THÀNH CÔNG!')",
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


def generate_v3_for(model_id: str, model_name: str, config_file: str, nb_name: str):
    payload = build_repo_payload()
    nb = make_v3_notebook(model_id, model_name, config_file, payload)
    nb_str = json.dumps(nb, indent=1, ensure_ascii=False)

    # 1. Ghi vào notebooks/
    out_path = Path("notebooks") / nb_name
    out_path.write_text(nb_str, encoding="utf-8")
    print(f"Đã tạo notebook {model_id} v3: {out_path} ({len(nb['cells'])} cells)")

    # 2. Ghi vào kaggle_deploy/<mid>_v3
    deploy_dir = Path("kaggle_deploy") / f"{model_id.lower()}_v3"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / nb_name).write_text(nb_str, encoding="utf-8")

    slug = f"bcb-{model_id.lower()}-{model_name.lower()}-512-v3"
    title = f"BCB {model_id} {model_name} 512 v3"
    meta = {
        "id": f"kichirou/{slug}",
        "title": title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            "kichirou/semanticcodeclonedetection-dataset-m4"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    (deploy_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Đã chuẩn bị thư mục deploy: {deploy_dir} (kernel: {slug})")


def main():
    generate_v3_for("M1", "CodeBERT", "configs/exp002_codebert_bcb.yaml", "bcb_exp-bcb-002_codebert_512_v3.ipynb")
    generate_v3_for("M2", "GraphCodeBERT", "configs/exp003_graphcodebert_bcb.yaml", "bcb_exp-bcb-003_graphcodebert_512_v3.ipynb")


if __name__ == "__main__":
    main()
