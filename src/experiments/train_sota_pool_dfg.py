"""Huấn luyện Nâng cao cho M1 (CodeBERT) và M2 (GraphCodeBERT):
1. Pooling (Mean+Max+CLS) kết hợp Supervised Contrastive (SupCon) & Focal Loss
2. True Data Flow Graph (DFG) với Graph-Guided Attention Mask cho GraphCodeBERT
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import sys
import tarfile
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sota_pool_dfg")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_file(filename: str) -> Path | None:
    search_paths = [
        Path(f"results/{filename}"),
        Path(f"data/{filename}"),
        Path(f"data/raw/bigclonebench/{filename}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/{filename}"),
        Path(f"/kaggle/input/semanticcodeclonedetection-dataset-m4/raw/bigclonebench/{filename}"),
        Path(f"/kaggle/working/{filename}"),
        Path(f"/kaggle/working/data/{filename}"),
    ]
    for p in search_paths:
        if p.exists():
            return p
    for root in [Path("results"), Path("data"), Path("/kaggle/input"), Path("/kaggle/working")]:
        if root.exists():
            for sub in root.rglob(filename):
                if sub.exists() and sub.is_file():
                    return sub
    return None


# =========================================================================
# LOSS FUNCTIONS: FOCAL LOSS & SUPERVISED CONTRASTIVE (SUPCON)
# =========================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        return (focal_weight * bce_loss).mean()


def supervised_contrastive_loss(
    emb_a: torch.Tensor, emb_b: torch.Tensor, targets: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    norm_a = F.normalize(emb_a, p=2, dim=-1)
    norm_b = F.normalize(emb_b, p=2, dim=-1)
    cos_sim = (norm_a * norm_b).sum(dim=-1)

    # Clone pairs (targets == 1): maximize cosine similarity
    loss_pos = 1.0 - cos_sim
    # Non-clone pairs (targets == 0): push cosine similarity below margin
    loss_neg = torch.clamp(cos_sim - margin, min=0.0) ** 2

    loss = targets * loss_pos + (1.0 - targets) * loss_neg
    return loss.mean()


# =========================================================================
# MODEL 1 & 2: POOLING (MEAN + MAX + CLS) + CONTRASTIVE HEAD
# =========================================================================
class TransformerPoolingClassifier(nn.Module):
    def __init__(self, hf_id: str, dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_side(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        cls_token = last_hidden[:, 0, :]
        sum_hidden = (last_hidden * mask).sum(dim=1)
        mean_hidden = sum_hidden / mask.sum(dim=1).clamp(min=1e-9)

        masked_hidden = last_hidden.masked_fill(mask == 0, -1e9)
        max_hidden = masked_hidden.max(dim=1).values

        combined = torch.cat([cls_token, mean_hidden, max_hidden], dim=-1)
        emb = self.proj(combined)
        return emb

    def forward(
        self,
        in_a: Dict[str, torch.Tensor],
        in_b: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ea = self.encode_side(in_a["input_ids"], in_a["attention_mask"])
        eb = self.encode_side(in_b["input_ids"], in_b["attention_mask"])

        diff = torch.abs(ea - eb)
        mult = ea * eb
        pair_rep = torch.cat([ea, eb, diff, mult], dim=-1)
        logits = self.classifier(pair_rep).squeeze(-1)
        return logits, ea, eb


# =========================================================================
# MODEL 3: TRUE DATA FLOW GRAPH (DFG) EXTRACTOR & GRAPHCODEBERT
# =========================================================================
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float",
    "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native",
    "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp",
    "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void",
    "volatile", "while", "true", "false", "null", "String", "System", "out", "println"
}


def extract_dfg_java(code_str: str, max_vars: int = 32) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Trích xuất danh sách biến và luồng dữ liệu (Data Flow Graph) từ Java method."""
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_java as tsjava

        lang = Language(tsjava.language())
        parser = Parser(lang)
        tree = parser.parse(bytes(code_str, "utf8"))

        vars_found: List[str] = []
        var_to_id: Dict[str, List[int]] = {}
        edges: Set[Tuple[int, int]] = set()

        def traverse(node):
            if node.type == "identifier":
                name = code_str[node.start_byte : node.end_byte]
                if name not in JAVA_KEYWORDS and len(name) > 1 and len(vars_found) < max_vars:
                    curr_id = len(vars_found)
                    vars_found.append(name)
                    if name in var_to_id:
                        prev_id = var_to_id[name][-1]
                        edges.add((prev_id, curr_id))
                        var_to_id[name].append(curr_id)
                    else:
                        var_to_id[name] = [curr_id]
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return vars_found, list(edges)
    except Exception:
        # Dự phòng bằng regex an toàn nếu tree-sitter gặp lỗi cú pháp
        tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_str)
        vars_found = []
        edges = []
        seen = {}
        for tok in tokens:
            if tok not in JAVA_KEYWORDS and len(tok) > 1 and len(vars_found) < max_vars:
                curr_id = len(vars_found)
                vars_found.append(tok)
                if tok in seen:
                    edges.append((seen[tok], curr_id))
                seen[tok] = curr_id
        return vars_found, edges


def build_dfg_features(
    code_str: str, tokenizer: Any, max_length: int = 512, max_vars: int = 32
) -> Tuple[List[int], List[int], torch.Tensor]:
    """Xây dựng chuỗi tokens kèm DFG nodes và ma trận Graph-Guided Attention Mask."""
    vars_found, edges = extract_dfg_java(code_str, max_vars=max_vars)

    code_tokens = tokenizer.tokenize(code_str)[: max_length - len(vars_found) - 3]
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]

    var_token_ids = []
    for v in vars_found:
        v_ids = tokenizer.tokenize(v)
        tokens.append(v_ids[0] if v_ids else tokenizer.unk_token)
        var_token_ids.append(len(tokens) - 1)
    tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    seq_len = len(input_ids)

    # Graph-guided attention matrix: [seq_len, seq_len]
    attn_mask = torch.zeros((seq_len, seq_len), dtype=torch.float32)

    # Code tokens attend to all code tokens
    code_len = len(code_tokens) + 2
    attn_mask[:code_len, :code_len] = 1.0

    # Var tokens attend along DFG edges and to itself
    for i in range(len(vars_found)):
        idx_i = code_len + i
        attn_mask[idx_i, idx_i] = 1.0
        # Attend to corresponding code tokens if name matches
        v_name = vars_found[i]
        for c_idx, ct in enumerate(code_tokens):
            if v_name in ct:
                attn_mask[idx_i, c_idx + 1] = 1.0
                attn_mask[c_idx + 1, idx_i] = 1.0

    for u, v in edges:
        if u < len(vars_found) and v < len(vars_found):
            idx_u = code_len + u
            idx_v = code_len + v
            attn_mask[idx_u, idx_v] = 1.0
            attn_mask[idx_v, idx_u] = 1.0

    # SEP token
    attn_mask[-1, :] = 1.0
    attn_mask[:, -1] = 1.0

    return input_ids, [1] * seq_len, attn_mask


class TrueDFGGraphCodeBERT(nn.Module):
    def __init__(self, hf_id: str = "microsoft/graphcodebert-base", dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_dfg(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask can be 2D [B, L] or 3D [B, L, L] for graph-guided attention
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]

        cls_rep = last_hidden[:, 0, :]
        if attention_mask.dim() == 2:
            mask = attention_mask.unsqueeze(-1).float()
            mean_rep = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            # 3D attention mask
            diag_mask = torch.diagonal(attention_mask, dim1=-2, dim2=-1).unsqueeze(-1).float()
            mean_rep = (last_hidden * diag_mask).sum(dim=1) / diag_mask.sum(dim=1).clamp(min=1e-9)

        rep = self.proj(torch.cat([cls_rep, mean_rep], dim=-1))
        return rep

    def forward(
        self,
        in_a: Dict[str, torch.Tensor],
        in_b: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ea = self.encode_dfg(in_a["input_ids"], in_a["attention_mask"])
        eb = self.encode_dfg(in_b["input_ids"], in_b["attention_mask"])

        diff = torch.abs(ea - eb)
        mult = ea * eb
        pair_rep = torch.cat([ea, eb, diff, mult], dim=-1)
        logits = self.classifier(pair_rep).squeeze(-1)
        return logits, ea, eb


# =========================================================================
# EVALUATION & THRESHOLD OPTIMIZATION
# =========================================================================
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for in_a, in_b, labels in loader:
            in_a = {k: v.to(device) for k, v in in_a.items()}
            in_b = {k: v.to(device) for k, v in in_b.items()}
            with autocast(enabled=torch.cuda.is_available()):
                logits, _, _ = model(in_a, in_b)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    best_f1 = -1.0
    best_tau = 0.5
    best_metrics = {}

    for tau in np.linspace(0.05, 0.95, 91):
        preds = (probs >= tau).astype(int)
        y = labels.astype(int)
        f1 = float(f1_score(y, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(round(tau, 2))
            acc = float(accuracy_score(y, preds))
            prec = float(precision_score(y, preds, zero_division=0))
            rec = float(recall_score(y, preds, zero_division=0))
            tp = int(((preds == 1) & (y == 1)).sum())
            tn = int(((preds == 0) & (y == 0)).sum())
            fp = int(((preds == 1) & (y == 0)).sum())
            fn = int(((preds == 0) & (y == 1)).sum())
            try:
                auc = float(roc_auc_score(y, probs))
            except Exception:
                auc = 0.0

            best_metrics = {
                "threshold": best_tau,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "roc_auc": round(auc, 4),
            }
    return best_tau, best_metrics


# =========================================================================
# DATASET & COLLATORS
# =========================================================================
class CodePairSimpleDataset(Dataset):
    def __init__(self, pairs: List[Dict[str, Any]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        return {
            "func1": str(p["func1"]),
            "func2": str(p["func2"]),
            "label": float(p["label"]),
        }


def make_standard_collate_fn(tokenizer: Any, max_length: int = 512):
    def collate_fn(batch: List[Dict[str, Any]]):
        f1 = [b["func1"] for b in batch]
        f2 = [b["func2"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        enc_a = tokenizer(f1, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc_b = tokenizer(f2, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        return enc_a, enc_b, labels
    return collate_fn


def make_dfg_collate_fn(tokenizer: Any, max_length: int = 512):
    def collate_fn(batch: List[Dict[str, Any]]):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        # Batch encode with DFG features
        b_input_a, b_mask_a = [], []
        b_input_b, b_mask_b = [], []

        for b in batch:
            ids_a, mask_a, _ = build_dfg_features(b["func1"], tokenizer, max_length=max_length)
            ids_b, mask_b, _ = build_dfg_features(b["func2"], tokenizer, max_length=max_length)
            b_input_a.append(ids_a)
            b_mask_a.append(mask_a)
            b_input_b.append(ids_b)
            b_mask_b.append(mask_b)

        # Pad to max length in batch
        def pad_side(id_lists, mask_lists):
            max_l = max(len(l) for l in id_lists)
            pad_ids = [l + [tokenizer.pad_token_id] * (max_l - len(l)) for l in id_lists]
            pad_masks = [l + [0] * (max_l - len(l)) for l in mask_lists]
            return {
                "input_ids": torch.tensor(pad_ids, dtype=torch.long),
                "attention_mask": torch.tensor(pad_masks, dtype=torch.long),
            }

        enc_a = pad_side(b_input_a, b_mask_a)
        enc_b = pad_side(b_input_b, b_mask_b)
        return enc_a, enc_b, labels
    return collate_fn


# =========================================================================
# GENERAL TRAINING LOOP
# =========================================================================
def train_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epochs: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 0.05,
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(">>> BẮT ĐẦU HUẤN LUYỆN: %s <<<", model_name)
    logger.info("=" * 80)

    model.to(device)
    focal_criterion = FocalLoss(gamma=2.0, alpha=0.25)

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    best_val_f1 = -1.0
    checkpoint_path = output_dir / f"best_{model_name.lower()}.pt"
    history = []
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        ep_t0 = time.perf_counter()

        for step, (in_a, in_b, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            in_a = {k: v.to(device) for k, v in in_a.items()}
            in_b = {k: v.to(device) for k, v in in_b.items()}
            labels = labels.to(device)

            with autocast(enabled=torch.cuda.is_available()):
                logits, ea, eb = model(in_a, in_b)
                loss_focal = focal_criterion(logits, labels)
                loss_supcon = supervised_contrastive_loss(ea, eb, labels, margin=0.2)
                loss = loss_focal + 0.3 * loss_supcon

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if step % 100 == 0 or step == len(train_loader):
                logger.info(
                    "[%s] Ep %d/%d | Step %d/%d | Loss: %.4f (Focal: %.4f, SupCon: %.4f) | LR: %.2e",
                    model_name, epoch, epochs, step, len(train_loader),
                    train_loss / n_batches, loss_focal.item(), loss_supcon.item(),
                    scheduler.get_last_lr()[0],
                )

        ep_duration = time.perf_counter() - ep_t0
        avg_train_loss = train_loss / n_batches

        # Validation
        val_probs, val_labels = evaluate_model(model, val_loader, device)
        tau_val, val_metrics = find_best_threshold(val_probs, val_labels)

        logger.info(
            "[%s] Epoch %d Kết thúc (%.1fs) | Loss: %.4f | Val F1: %.4f (tau=%.2f) | Val Acc: %.4f | Val AUC: %.4f",
            model_name, epoch, ep_duration, avg_train_loss,
            val_metrics["f1"], tau_val, val_metrics["accuracy"], val_metrics["roc_auc"],
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "time_sec": ep_duration,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(">>> Đã lưu Best Checkpoint tại Epoch %d (Val F1: %.4f) <<<", epoch, best_val_f1)

    total_time_h = (time.perf_counter() - t_start) / 3600.0
    logger.info("[%s] Huấn luyện hoàn thành sau %.2f giờ.", model_name, total_time_h)

    # Đánh giá trên tập test 5.000 mẫu
    logger.info(">>> Đang đánh giá %s trên tập kiểm thử chuẩn 5.000 mẫu BigCloneBench...", model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_probs, test_labels = evaluate_model(model, test_loader, device)
    best_tau, test_metrics = find_best_threshold(test_probs, test_labels)
    test_metrics["train_time_hours"] = round(total_time_h, 2)
    test_metrics["model_name"] = model_name

    logger.info("=" * 80)
    logger.info("[%s TEST RESULTS] tau=%.2f | F1=%.4f | Acc=%.4f | Prec=%.4f | Rec=%.4f | AUC=%.4f",
                model_name, best_tau, test_metrics["f1"], test_metrics["accuracy"],
                test_metrics["precision"], test_metrics["recall"], test_metrics["roc_auc"])
    logger.info("[%s CONFUSION MATRIX] TP=%d | TN=%d | FP=%d | FN=%d",
                model_name, test_metrics["tp"], test_metrics["tn"], test_metrics["fp"], test_metrics["fn"])
    logger.info("=" * 80)

    # Lưu kết quả
    np.save(output_dir / f"{model_name.lower()}_test_probs.npy", test_probs)
    with open(output_dir / f"{model_name.lower()}_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    with open(output_dir / f"{model_name.lower()}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return test_metrics


# =========================================================================
# DATA LOADING
# =========================================================================
def load_datasets(n_train: int = 30000, n_val: int = 2000):
    test_file = locate_file("bcb_5k_test_pairs.jsonl")
    if not test_file:
        raise FileNotFoundError("Không tìm thấy bcb_5k_test_pairs.jsonl!")
    test_pairs = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line))
    logger.info("Đã nạp %d cặp test chuẩn từ %s", len(test_pairs), test_file)

    train_file = locate_file("train_shard_0.parquet")
    df_train = pd.read_parquet(train_file)
    half_train = n_train // 2
    df_pos = df_train[df_train["label"] == True].sample(n=half_train, random_state=42)
    df_neg = df_train[df_train["label"] == False].sample(n=half_train, random_state=42)
    df_sampled = pd.concat([df_pos, df_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_pairs = df_sampled[["func1", "func2", "label"]].to_dict(orient="records")
    for p in train_pairs:
        p["label"] = 1 if p["label"] else 0
    logger.info("Đã nạp & cân bằng %d cặp train (15k clone, 15k non-clone)", len(train_pairs))

    val_file = locate_file("validation_shard_0.parquet")
    df_val = pd.read_parquet(val_file)
    half_val = n_val // 2
    df_vpos = df_val[df_val["label"] == True].sample(n=min(half_val, len(df_val[df_val["label"] == True])), random_state=42)
    df_vneg = df_val[df_val["label"] == False].sample(n=min(half_val, len(df_val[df_val["label"] == False])), random_state=42)
    df_v_sampled = pd.concat([df_vpos, df_vneg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_pairs = df_v_sampled[["func1", "func2", "label"]].to_dict(orient="records")
    for p in val_pairs:
        p["label"] = 1 if p["label"] else 0
    logger.info("Đã nạp & cân bằng %d cặp val", len(val_pairs))

    return train_pairs, val_pairs, test_pairs


# =========================================================================
# MAIN DRIVER
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Pooling SupCon & True DFG")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="results/exp_sota_pool_dfg")
    args = parser.parse_args()

    set_seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Thiết bị tính toán: %s (CUDA Available: %s)", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s | VRAM: %.1f GB", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / (1024**3))

    train_pairs, val_pairs, test_pairs = load_datasets()

    all_metrics = {}

    # ---------------------------------------------------------------------
    # 1. Model M1: CodeBERT với Pooling (Mean+Max+CLS) + Contrastive & Focal
    # ---------------------------------------------------------------------
    tok_m1 = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    coll_m1 = make_standard_collate_fn(tok_m1, max_length=512)
    tr_loader_m1 = DataLoader(CodePairSimpleDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=coll_m1)
    val_loader_m1 = DataLoader(CodePairSimpleDataset(val_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_m1)
    te_loader_m1 = DataLoader(CodePairSimpleDataset(test_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_m1)

    model_m1 = TransformerPoolingClassifier("microsoft/codebert-base", dropout=0.2)
    res_m1 = train_model("M1_CodeBERT_Pool_SupCon", model_m1, tr_loader_m1, val_loader_m1, te_loader_m1, device, out_dir, epochs=args.epochs, lr=args.lr)
    all_metrics["M1_CodeBERT_Pool_SupCon"] = res_m1
    del model_m1
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 2. Model M2: GraphCodeBERT với Pooling (Mean+Max+CLS) + Contrastive & Focal
    # ---------------------------------------------------------------------
    tok_m2 = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    coll_m2 = make_standard_collate_fn(tok_m2, max_length=512)
    tr_loader_m2 = DataLoader(CodePairSimpleDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=coll_m2)
    val_loader_m2 = DataLoader(CodePairSimpleDataset(val_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_m2)
    te_loader_m2 = DataLoader(CodePairSimpleDataset(test_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_m2)

    model_m2 = TransformerPoolingClassifier("microsoft/graphcodebert-base", dropout=0.3)
    res_m2 = train_model("M2_GraphCodeBERT_Pool_SupCon", model_m2, tr_loader_m2, val_loader_m2, te_loader_m2, device, out_dir, epochs=args.epochs, lr=args.lr)
    all_metrics["M2_GraphCodeBERT_Pool_SupCon"] = res_m2
    del model_m2
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # 3. Model M3: GraphCodeBERT với True Data Flow Graph (DFG)
    # ---------------------------------------------------------------------
    coll_dfg = make_dfg_collate_fn(tok_m2, max_length=512)
    tr_loader_dfg = DataLoader(CodePairSimpleDataset(train_pairs), batch_size=args.batch_size, shuffle=True, collate_fn=coll_dfg)
    val_loader_dfg = DataLoader(CodePairSimpleDataset(val_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_dfg)
    te_loader_dfg = DataLoader(CodePairSimpleDataset(test_pairs), batch_size=args.batch_size * 2, shuffle=False, collate_fn=coll_dfg)

    model_dfg = TrueDFGGraphCodeBERT("microsoft/graphcodebert-base", dropout=0.25)
    res_dfg = train_model("M2_GraphCodeBERT_True_DFG", model_dfg, tr_loader_dfg, val_loader_dfg, te_loader_dfg, device, out_dir, epochs=args.epochs, lr=args.lr)
    all_metrics["M2_GraphCodeBERT_True_DFG"] = res_dfg
    del model_dfg
    torch.cuda.empty_cache()

    # Tổng kết & Đóng gói
    summary_file = out_dir / "all_sota_pool_dfg_test_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(">>> ĐÃ LƯU TOÀN BỘ KẾT QUẢ TẠI %s <<<", summary_file)

    bundle_path = out_dir / "bcb_sota_pool_dfg_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tar:
        for f in out_dir.iterdir():
            if f.name != bundle_path.name:
                tar.add(f, arcname=f.name)
    logger.info(">>> ĐÃ ĐÓNG GÓI BUNDLE: %s (%.2f MB) <<<", bundle_path, bundle_path.stat().st_size / (1024 * 1024))


if __name__ == "__main__":
    main()
