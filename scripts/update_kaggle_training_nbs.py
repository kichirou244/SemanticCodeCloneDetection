"""Cập nhật các notebook huấn luyện M3 và M4 để tự động tận dụng Kaggle Dataset đặc trưng đã trích xuất."""

import json
from pathlib import Path

configs = [
    (
        "notebooks/bcb_exp-bcb-004_astnn_kaggle.ipynb",
        [
            "# Chuẩn bị dữ liệu BigCloneBench (tải tập smoke và dev qua streaming)\n",
            "import os\n",
            "if os.path.exists('/kaggle/input/bigclonebench-features/bcb_ast_cache.npz'):\n",
            "    print('Phát hiện Kaggle Dataset: Đã có sẵn AST cache!')\n",
            "    os.makedirs('artifacts/bcb_features', exist_ok=True)\n",
            "    sh('cp /kaggle/input/bigclonebench-features/bcb_ast_cache.npz artifacts/bcb_features/')\n",
            "sh('python scripts/download_bcb_subsets.py --smoke --dev')\n",
            "sh('ls -lh data/raw/bigclonebench/')\n",
        ],
    ),
    (
        "notebooks/bcb_exp-bcb-005_cpg_gnn_kaggle.ipynb",
        [
            "# Chuẩn bị dữ liệu BigCloneBench (tải tập smoke và dev qua streaming)\n",
            "import os\n",
            "if os.path.exists('/kaggle/input/bigclonebench-features/bcb_cpg_export.tar.gz'):\n",
            "    print('Phát hiện Kaggle Dataset: Đã có sẵn CPG cache!')\n",
            "    os.makedirs('artifacts', exist_ok=True)\n",
            "    sh('tar -xzf /kaggle/input/bigclonebench-features/bcb_cpg_export.tar.gz -C artifacts/')\n",
            "sh('python scripts/download_bcb_subsets.py --smoke --dev')\n",
            "sh('ls -lh data/raw/bigclonebench/')\n",
        ],
    ),
]

for nb_path_str, feat_source in configs:
    nb_path = Path(nb_path_str)
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if any("download_bcb_subsets.py" in line for line in cell.get("source", [])):
            cell["source"] = feat_source
    with nb_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Đã cập nhật:", nb_path)
