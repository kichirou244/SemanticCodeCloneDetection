"""Tự động sinh bộ Kaggle Notebooks hoàn chỉnh cho BigCloneBench (M1..M4)."""

import json
from pathlib import Path


def make_notebook(exp_id: str, exp_name: str, config_file: str, extra_pip: list[str] = None) -> dict:
    extra_installs = " ".join(extra_pip) if extra_pip else ""
    pip_cmd = f"pip install -q transformers datasets scikit-learn tree-sitter tree-sitter-java pyyaml {extra_installs}".strip()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {exp_id} — {exp_name} trên BigCloneBench (Java) - Kaggle GPU\n",
                "\n",
                "**Setup Kaggle:** Settings → Accelerator = **GPU (T4/P100)**, Internet = **ON**.\n",
                "\n",
                "Chạy các cell theo thứ tự. Khi cần tiếp tục phiên bị ngắt, đổi `MODE = 'resume'` ở cell Train.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json, os, subprocess, sys\n",
                "\n",
                "GITHUB_TOKEN = ''\n",
                "HF_TOKEN = ''\n",
                "try:\n",
                "    from kaggle_secrets import UserSecretsClient\n",
                "    _s = UserSecretsClient()\n",
                "    GITHUB_TOKEN = GITHUB_TOKEN or _s.get_secret('GITHUB_TOKEN')\n",
                "    HF_TOKEN = HF_TOKEN or _s.get_secret('HF_TOKEN')\n",
                "except Exception as e:\n",
                "    print('Dùng token dán trực tiếp nếu có:', e)\n",
                "if HF_TOKEN:\n",
                "    os.environ['HF_TOKEN'] = HF_TOKEN\n",
                "print('GITHUB_TOKEN:', 'OK' if GITHUB_TOKEN else 'TRỐNG', '| HF_TOKEN:', 'OK' if HF_TOKEN else 'TRỐNG')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "REPO = 'kichirou244/SemanticCodeCloneDetection'\n",
                "\n",
                "def sh(cmd, **kw):\n",
                "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)\n",
                "    print(f'$ {cmd}\\n{r.stdout}{r.stderr}')\n",
                "    if r.returncode != 0:\n",
                "        raise RuntimeError(f'Lệnh thất bại (rc={r.returncode}): {cmd}')\n",
                "    return r.stdout\n",
                "\n",
                "if os.path.isdir('project'):\n",
                "    print('project đã tồn tại -> git pull')\n",
                "    sh('git -C project pull --ff-only')\n",
                "else:\n",
                "    url = f'https://{GITHUB_TOKEN}@github.com/{REPO}.git' if GITHUB_TOKEN else f'https://github.com/{REPO}.git'\n",
                "    sh(f'git clone \"{url}\" project')\n",
                "sh(f'git -C project remote set-url origin https://github.com/{REPO}.git')\n",
                "os.chdir('project')\n",
                "print('cwd =', os.getcwd())\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"sh('{pip_cmd}')\n",
                "import torch\n",
                "assert torch.cuda.is_available(), 'Cần bật GPU trong Settings -> Accelerator -> GPU'\n",
                "print('GPU:', torch.cuda.get_device_name(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Chuẩn bị dữ liệu BigCloneBench (tải tập smoke và dev qua streaming)\n",
                "sh('python scripts/download_bcb_subsets.py --smoke --dev')\n",
                "sh('ls -lh data/raw/bigclonebench/')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Preflight check: kiểm tra cấu hình {config_file}\n",
                f"sh('python -m src.experiments.bcb_runner --config {config_file} --mode preflight --device cuda')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"# Diagnostic check: kiểm tra 1 batch loss giảm\n",
                f"sh('python -m src.experiments.bcb_runner --config {config_file} --mode diagnostic --device cuda')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Huấn luyện mô hình (hỗ trợ resume và time budget)\n",
                "MODE = 'train'  # 'train' hoặc 'resume'\n",
                "TIME_BUDGET_S = 9 * 3600  # Giới hạn 9h an toàn\n",
                f"sh(f'python -m src.experiments.bcb_runner --config {config_file} --mode {{MODE}} --device cuda --time-budget-s {{TIME_BUDGET_S}}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Đánh giá trên tập test và chốt metrics\n",
                f"sh('python -m src.experiments.bcb_runner --config {config_file} --mode evaluate --device cuda')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Đóng gói kết quả metrics và predictions để tải về (dung lượng nhẹ vài MB)\n",
                f"sh('tar -czf {exp_id.lower()}_results.tar.gz results/')\n",
                f"sh('ls -lh {exp_id.lower()}_results.tar.gz')\n",
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
    out_dir = Path("notebooks")
    out_dir.mkdir(parents=True, exist_ok=True)

    notebooks = [
        ("EXP-BCB-002", "CodeBERT", "configs/exp002_codebert_bcb.yaml", []),
        ("EXP-BCB-003", "GraphCodeBERT", "configs/exp003_graphcodebert_bcb.yaml", []),
        ("EXP-BCB-004", "ASTNN", "configs/exp004_astnn_bcb.yaml", []),
        ("EXP-BCB-005", "CPG-GNN", "configs/exp005_cpg_gnn_bcb.yaml", ["torch-geometric"]),
    ]

    for exp_id, name, cfg, extra in notebooks:
        nb = make_notebook(exp_id, name, cfg, extra)
        fname = f"bcb_{exp_id.lower()}_{name.lower().replace('-', '_')}_kaggle.ipynb"
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
