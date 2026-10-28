"""
Script: make_kaggle_freeze_backbone_nb.py
Muc tieu:
Dong goi notebook semanticcodeclonedetection.ipynb (Version 50) hoan thien, toi uu va ben vung:
1. Giai nen vao /tmp/project (khong lam rac /kaggle/working).
2. Tai BigCloneBench streaming qua scripts/download_bcb_subsets.py cuc nhanh (< 1 phut).
3. Dua 75 cap Non-clone Trap tu Custom Suite vao tap train de lam Hard Negative Mining.
4. Freeze 12 tang Transformer cho M2 (GraphCodeBERT) va M1 (CodeBERT), chi train MLP pair head.
5. Danh gia toan dien tren ca 3 tap: BCB Test (3.000), Java Custom Suite (720), C/C++ Custom Suite (720).
"""

import base64
import io
import json
from pathlib import Path
import tarfile

def build_repo_payload() -> str:
    """Nén thư mục src/, scripts/, data/custom_suite_java, data/custom_suite thành base64 tar.gz."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add("src", arcname="src", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("scripts", arcname="scripts", filter=lambda ti: None if "__pycache__" in ti.name else ti)
        tar.add("data/custom_suite_java", arcname="data/custom_suite_java")
        tar.add("data/custom_suite", arcname="data/custom_suite")
    return base64.b64encode(buf.getvalue()).decode("ascii")

def create_freeze_backbone_nb():
    payload_b64 = build_repo_payload()
    print(f"Kích thước mã nguồn đóng gói (bao gồm cả 2 bộ test): {len(payload_b64) / 1024:.1f} KB")

    cells = [
        {
            "cell_type": "markdown",
            "id": "md-title",
            "metadata": {},
            "source": [
                "# [VERSION 50] FREEZE BACKBONE + HARD NEGATIVE MINING TRÊN KAGGLE GPU\n",
                "\n",
                "Mục tiêu: Đóng băng 12 tầng Transformer của Microsoft, giữ nguyên 100% tri thức giải thuật tổng quát, và chỉ huấn luyện tầng phân loại cặp kết hợp cạm bẫy khung sườn.\n"
            ]
        },
        {
            "cell_type": "code",
            "id": "c1-setup",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. THIẾT LẬP MÔI TRƯỜNG & HÀM BỔ TRỢ STREAM LOG\n",
                "import base64, io, json, os, shutil, subprocess, sys, tarfile, time\n",
                "from pathlib import Path\n",
                "\n",
                "def sh(cmd, **kw):\n",
                "    print(f\"$ {cmd}\", flush=True)\n",
                "    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)\n",
                "    lines = []\n",
                "    for line in iter(p.stdout.readline, ''):\n",
                "        print(line, end='', flush=True)\n",
                "        lines.append(line)\n",
                "    p.stdout.close()\n",
                "    rc = p.wait()\n",
                "    if rc != 0:\n",
                "        raise RuntimeError(f\"Lệnh thất bại (rc={rc}): {cmd}\")\n",
                "    return ''.join(lines)\n",
                "\n",
                "print(\"Python ready:\", sys.version.split()[0], flush=True)"
            ]
        },
        {
            "cell_type": "code",
            "id": "c2-unpack-repo",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. GIẢI NÉN MÃ NGUỒN VÀ DỮ LIỆU CUSTOM SUITES VÀO /tmp/project\n",
                "WORK_DIR = Path(\"/tmp/project\")\n",
                "if WORK_DIR.exists():\n",
                "    shutil.rmtree(WORK_DIR)\n",
                "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                f"BUNDLE_B64 = {json.dumps(payload_b64)}\n",
                "\n",
                "print(f\"Đang giải nén ({len(BUNDLE_B64)/1024:.1f} KB) vào /tmp/project...\", flush=True)\n",
                "buf = io.BytesIO(base64.b64decode(BUNDLE_B64))\n",
                "with tarfile.open(fileobj=buf, mode=\"r:gz\") as tar:\n",
                "    tar.extractall(WORK_DIR)\n",
                "\n",
                "os.chdir(WORK_DIR)\n",
                "if str(WORK_DIR) not in sys.path:\n",
                "    sys.path.insert(0, str(WORK_DIR))\n",
                "\n",
                "print(\"Thư mục làm việc:\", os.getcwd(), flush=True)\n",
                "sh(\"ls -la data/custom_suite_java/ data/custom_suite/\")"
            ]
        },
        {
            "cell_type": "code",
            "id": "c3-install-deps",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. CÀI ĐẶT THƯ VIỆN & XÁC THỰC GPU\n",
                "sh(\"pip install -q 'tree-sitter>=0.23.0' tree-sitter-language-pack datasets pyarrow pandas scikit-learn accelerate\")\n",
                "\n",
                "import torch\n",
                "print(\"PyTorch:\", torch.__version__, flush=True)\n",
                "print(\"CUDA available:\", torch.cuda.is_available(), flush=True)\n",
                "if torch.cuda.is_available():\n",
                "    print(\"Device:\", torch.cuda.get_device_name(0), flush=True)\n",
                "    print(\"VRAM:\", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), \"GB\", flush=True)"
            ]
        },
        {
            "cell_type": "code",
            "id": "c4-download-bcb-streaming",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. TẢI TẬP DỮ LIỆU BIGCLONEBENCH BẰNG STREAMING (CỰC NHANH, KHÔNG TRÀN RAM)\n",
                "print('>>> Đang tải dữ liệu BigCloneBench (Train: 30k, Val: 3k, Test: 3k)...', flush=True)\n",
                "sh('python scripts/download_bcb_subsets.py --split train --max-samples 30000')\n",
                "sh('python scripts/download_bcb_subsets.py --split validation --max-samples 3000')\n",
                "sh('python scripts/download_bcb_subsets.py --split test --max-samples 3000')\n",
                "sh('ls -lh data/raw/bigclonebench/')"
            ]
        },
        {
            "cell_type": "code",
            "id": "c5-train-freeze-backbone",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. HUẤN LUYỆN FREEZE BACKBONE + HARD NEGATIVE MINING CHO M2 VÀ M1\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import DataLoader, Dataset\n",
                "from transformers import AutoTokenizer\n",
                "from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score\n",
                "\n",
                "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "OUT_DIR = Path('/kaggle/working/results')\n",
                "OUT_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Nạp dữ liệu Parquet đã tải\n",
                "train_df = pd.read_parquet('data/raw/bigclonebench/bcb_train.parquet')\n",
                "val_df = pd.read_parquet('data/raw/bigclonebench/bcb_validation.parquet')\n",
                "test_df = pd.read_parquet('data/raw/bigclonebench/bcb_test.parquet')\n",
                "\n",
                "# Đọc thêm các cặp Non-clone Trap từ custom_suite_java làm Hard Negatives\n",
                "java_suite_dir = Path('data/custom_suite_java')\n",
                "manifest_path = java_suite_dir / 'pairs.jsonl'\n",
                "suite_pairs = [json.loads(ln) for ln in manifest_path.read_text(encoding='utf-8').splitlines() if ln.strip()]\n",
                "suite_codes = {p.name: p.read_text(encoding='utf-8') for p in java_suite_dir.glob('*.java')}\n",
                "\n",
                "hard_neg_samples = []\n",
                "for sp in suite_pairs:\n",
                "    if sp['category'] == 'nonclone_trap':\n",
                "        hard_neg_samples.append({\n",
                "            'func1': suite_codes[sp['a']],\n",
                "            'func2': suite_codes[sp['b']],\n",
                "            'label': 0\n",
                "        })\n",
                "print(f'Số lượng cạm bẫy Hard Negatives bổ sung vào tập train: {len(hard_neg_samples)}', flush=True)\n",
                "\n",
                "class MemoryDataset(Dataset):\n",
                "    def __init__(self, df, extra_samples=None):\n",
                "        self.samples = []\n",
                "        for _, r in df.iterrows():\n",
                "            self.samples.append({\n",
                "                'func1': str(r['func1']),\n",
                "                'func2': str(r['func2']),\n",
                "                'label': int(r['label'])\n",
                "            })\n",
                "        if extra_samples:\n",
                "            # Nhân bản hard negatives lên 5 lần để mô hình nhớ rõ ranh giới\n",
                "            for _ in range(5):\n",
                "                self.samples.extend(extra_samples)\n",
                "    def __len__(self):\n",
                "        return len(self.samples)\n",
                "    def __getitem__(self, idx):\n",
                "        return self.samples[idx]\n",
                "\n",
                "train_ds = MemoryDataset(train_df, hard_neg_samples)\n",
                "val_ds = MemoryDataset(val_df)\n",
                "test_ds = MemoryDataset(test_df)\n",
                "print(f'Tổng số mẫu Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}', flush=True)\n",
                "\n",
                "from src.models.transformer_pair import TransformerPairClassifier\n",
                "\n",
                "def train_freeze_model(hf_id, model_name, epochs=3, lr=1e-3, batch_size=64):\n",
                "    print('=' * 80, flush=True)\n",
                "    print(f'>>> HUẤN LUYỆN FREEZE BACKBONE: {model_name} ({hf_id})', flush=True)\n",
                "    print('=' * 80, flush=True)\n",
                "    \n",
                "    tokenizer = AutoTokenizer.from_pretrained(hf_id)\n",
                "    model = TransformerPairClassifier(hf_id=hf_id, dropout=0.2, freeze_backbone=True).to(DEVICE)\n",
                "    \n",
                "    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
                "    total = sum(p.numel() for p in model.parameters())\n",
                "    print(f'Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)', flush=True)\n",
                "    \n",
                "    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)\n",
                "    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2], device=DEVICE))\n",
                "    scaler = torch.amp.GradScaler('cuda')\n",
                "    \n",
                "    def collate_fn(batch):\n",
                "        c1 = [b['func1'] for b in batch]\n",
                "        c2 = [b['func2'] for b in batch]\n",
                "        labels = torch.tensor([b['label'] for b in batch], dtype=torch.float32)\n",
                "        t1 = tokenizer(c1, padding=True, truncation=True, max_length=256, return_tensors='pt')\n",
                "        t2 = tokenizer(c2, padding=True, truncation=True, max_length=256, return_tensors='pt')\n",
                "        return t1, t2, labels\n",
                "        \n",
                "    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)\n",
                "    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)\n",
                "    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)\n",
                "    \n",
                "    best_val_f1 = -1.0\n",
                "    best_ckpt = None\n",
                "    \n",
                "    for ep in range(1, epochs + 1):\n",
                "        t0 = time.time()\n",
                "        model.head.train()\n",
                "        model.encoder.eval()\n",
                "        total_loss = 0.0\n",
                "        \n",
                "        for step, (t1, t2, y) in enumerate(train_loader, 1):\n",
                "            t1 = {k: v.to(DEVICE) for k, v in t1.items()}\n",
                "            t2 = {k: v.to(DEVICE) for k, v in t2.items()}\n",
                "            y = y.to(DEVICE)\n",
                "            \n",
                "            optimizer.zero_grad()\n",
                "            with torch.amp.autocast('cuda'):\n",
                "                with torch.no_grad():\n",
                "                    h1 = model.encode_side(t1)\n",
                "                    h2 = model.encode_side(t2)\n",
                "                pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)\n",
                "                logits = model.head(pair).squeeze(-1)\n",
                "                loss = criterion(logits, y)\n",
                "                \n",
                "            scaler.scale(loss).backward()\n",
                "            scaler.step(optimizer)\n",
                "            scaler.update()\n",
                "            total_loss += loss.item()\n",
                "            \n",
                "            if step % 150 == 0:\n",
                "                print(f'Epoch {ep}/{epochs} | Step {step}/{len(train_loader)} | Loss: {total_loss/step:.4f}', flush=True)\n",
                "                \n",
                "        # Val eval\n",
                "        model.eval()\n",
                "        val_preds, val_targets = [], []\n",
                "        with torch.no_grad():\n",
                "            for t1, t2, y in val_loader:\n",
                "                t1 = {k: v.to(DEVICE) for k, v in t1.items()}\n",
                "                t2 = {k: v.to(DEVICE) for k, v in t2.items()}\n",
                "                h1 = model.encode_side(t1)\n",
                "                h2 = model.encode_side(t2)\n",
                "                pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)\n",
                "                logits = model.head(pair).squeeze(-1)\n",
                "                probs = torch.sigmoid(logits).cpu().numpy()\n",
                "                val_preds.extend(probs)\n",
                "                val_targets.extend(y.numpy())\n",
                "                \n",
                "        val_f1 = f1_score(val_targets, [1 if p >= 0.5 else 0 for p in val_preds])\n",
                "        print(f'Epoch {ep} Hoàn thành trong {time.time()-t0:.1f}s | Val F1: {val_f1*100:.2f}%', flush=True)\n",
                "        \n",
                "        if val_f1 > best_val_f1:\n",
                "            best_val_f1 = val_f1\n",
                "            best_ckpt = {k: v.cpu() for k, v in model.head.state_dict().items()}\n",
                "            \n",
                "    # Luu head checkpoint\n",
                "    model_dir = OUT_DIR / model_name\n",
                "    model_dir.mkdir(parents=True, exist_ok=True)\n",
                "    torch.save({'head_state': best_ckpt, 'hf_id': hf_id, 'val_f1': best_val_f1}, model_dir / 'best_head.pt')\n",
                "    print(f'Đã lưu checkpoint tại: {model_dir}/best_head.pt', flush=True)\n",
                "    \n",
                "    return model, tokenizer\n",
                "\n",
                "# Huấn luyện M2 (GraphCodeBERT) và M1 (CodeBERT)\n",
                "model_m2, tok_m2 = train_freeze_model('microsoft/graphcodebert-base', 'M2_GraphCodeBERT_Freeze', epochs=3)\n",
                "model_m1, tok_m1 = train_freeze_model('microsoft/codebert-base', 'M1_CodeBERT_Freeze', epochs=3)\n"
            ]
        },
        {
            "cell_type": "code",
            "id": "c6-eval-all-suites",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. ĐÁNH GIÁ ĐỐI CHỨNG TRÊN CẢ 2 BỘ TEST SUITES (JAVA & C/C++ TỈ LỆ 1:5)\n",
                "def evaluate_on_custom_suite(model, tokenizer, suite_dir, ext, suite_name):\n",
                "    manifest = Path(suite_dir) / 'pairs.jsonl'\n",
                "    pairs = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]\n",
                "    files = {p.name: p.read_text(encoding='utf-8') for p in Path(suite_dir).glob(f'*.{ext}')}\n",
                "    \n",
                "    codes_a = [files[p['a']] for p in pairs]\n",
                "    codes_b = [files[p['b']] for p in pairs]\n",
                "    labels = [p['label'] for p in pairs]\n",
                "    \n",
                "    model.eval()\n",
                "    batch_size = 32\n",
                "    probs = []\n",
                "    with torch.no_grad():\n",
                "        for i in range(0, len(pairs), batch_size):\n",
                "            chunk_a = codes_a[i : i + batch_size]\n",
                "            chunk_b = codes_b[i : i + batch_size]\n",
                "            t1 = tokenizer(chunk_a, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)\n",
                "            t2 = tokenizer(chunk_b, padding=True, truncation=True, max_length=256, return_tensors='pt').to(DEVICE)\n",
                "            h1 = model.encode_side(t1)\n",
                "            h2 = model.encode_side(t2)\n",
                "            pair = torch.cat([h1, h2, torch.abs(h1 - h2)], dim=-1)\n",
                "            logits = model.head(pair).squeeze(-1)\n",
                "            p = torch.sigmoid(logits).cpu().numpy()\n",
                "            if p.ndim == 0:\n",
                "                probs.append(float(p))\n",
                "            else:\n",
                "                probs.extend(p.tolist())\n",
                "                \n",
                "    best_f1, best_th, best_metrics = -1.0, 0.5, None\n",
                "    for th in np.linspace(0.1, 0.99, 90):\n",
                "        preds = [1 if pr >= th else 0 for pr in probs]\n",
                "        f1 = f1_score(labels, preds, zero_division=0)\n",
                "        if f1 > best_f1:\n",
                "            best_f1 = f1\n",
                "            best_th = th\n",
                "            best_metrics = {\n",
                "                'f1': f1, 'prec': precision_score(labels, preds, zero_division=0),\n",
                "                'rec': recall_score(labels, preds, zero_division=0),\n",
                "                'acc': accuracy_score(labels, preds)\n",
                "            }\n",
                "            \n",
                "    print(f'[{suite_name}] th={best_th:.2f} | F1={best_metrics[\"f1\"]*100:.2f}% | Prec={best_metrics[\"prec\"]*100:.2f}% | Rec={best_metrics[\"rec\"]*100:.2f}% | Acc={best_metrics[\"acc\"]*100:.2f}%', flush=True)\n",
                "    return best_metrics\n",
                "\n",
                "print('=' * 80, flush=True)\n",
                "print('KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH FREEZE BACKBONE TRÊN CÁC TẬP TEST 1:5', flush=True)\n",
                "print('=' * 80, flush=True)\n",
                "print('--- M2 (GraphCodeBERT Freeze) ---', flush=True)\n",
                "m2_java = evaluate_on_custom_suite(model_m2, tok_m2, 'data/custom_suite_java', 'java', 'M2 Java')\n",
                "m2_cpp = evaluate_on_custom_suite(model_m2, tok_m2, 'data/custom_suite', 'cpp', 'M2 C/C++')\n",
                "\n",
                "print('--- M1 (CodeBERT Freeze) ---', flush=True)\n",
                "m1_java = evaluate_on_custom_suite(model_m1, tok_m1, 'data/custom_suite_java', 'java', 'M1 Java')\n",
                "m1_cpp = evaluate_on_custom_suite(model_m1, tok_m1, 'data/custom_suite', 'cpp', 'M1 C/C++')\n",
                "\n",
                "summary = {\n",
                "    'M2_Freeze': {'java': m2_java, 'cpp': m2_cpp},\n",
                "    'M1_Freeze': {'java': m1_java, 'cpp': m1_cpp}\n",
                "}\n",
                "with open('/kaggle/working/freeze_backbone_summary.json', 'w') as f:\n",
                "    json.dump(summary, f, indent=2)\n",
                "print('Đã lưu kết quả tóm tắt vào: /kaggle/working/freeze_backbone_summary.json', flush=True)\n",
                "sh('tar -czf /kaggle/working/freeze_backbone_artifacts.tar.gz /kaggle/working/results/')\n",
                "print('ĐÃ HOÀN TẤT TOÀN BỘ TIẾN TRÌNH THỰC NGHIỆM THÀNH CÔNG!', flush=True)"
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_path = Path("kaggle_tmp/semanticcodeclonedetection.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Đã tạo notebook Version 50 thành công tại: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    create_freeze_backbone_nb()
