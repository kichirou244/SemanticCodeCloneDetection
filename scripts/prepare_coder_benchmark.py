#!/usr/bin/env python3
"""Chuan bi tap du lieu kiem thu CODER HUSC (C/C++) theo chuan BigCloneBench.

Quy trinh chuan hoa:
1. Doc ma nguon va metadata tu data/raw/coder/{problem_id}/code/.
2. Tien xu ly: loai bo comment, chuan hoa xuong dong, strip whitespace.
3. Tokenize & Chuẩn hóa Type-2 theo Svajlenko & Roy (ICSME 2014, 2016):
   - Thay ten bien/ham bang 'ID'
   - Thay hang so bang 'NUM'
   - Thay chuoi ky tu bang 'STR'
4. Do do tuong dong cu phap (Syntactic Similarity):
   Sim(f1, f2) = min(|T1 cap T2| / |T1|, |T1 cap T2| / |T2|)
5. Phan tang nhan theo chuan BigCloneBench:
   - Nhan 0 (Non-clone): Khac problem_id
   - Nhan 1 (Clone - Cung problem_id):
     * Type-1/2/ST3: Sim >= 0.70 (Nghi van sao chep / template)
     * MT3: 0.50 <= Sim < 0.70 (Tuong dong vua)
     * Type-4: Sim < 0.50 (Khac thuat toan / doc lap)
6. Xuat file data/processed/coder_benchmark/coder_test_pairs.jsonl
"""

from __future__ import annotations

from collections import Counter
import json
import logging
import os
from pathlib import Path
import random
import re
import sys

from pygments.lexers import CppLexer
from pygments.token import Token

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_coder_benchmark")

SEED = 42
random.seed(SEED)

lexer = CppLexer()


def clean_code(source: str) -> str:
    """Loai bo comments va chuan hoa khoang trang."""
    # Xoa comment block /* ... */
    cleaned = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    # Xoa comment inline // ...
    cleaned = re.sub(r"//.*", "", cleaned)
    # Chuan hoa xuong dong
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Loai bo cac dong trong lien tiep
    lines = [line.rstrip() for line in cleaned.split("\n")]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def tokenize_normalized(source: str) -> list[str]:
    """Tach tu va chuan hoa Type-2 theo chuan BigCloneBench."""
    tokens = []
    for ttype, val in lexer.get_tokens(source):
        if ttype in Token.Comment or ttype in Token.Text.Whitespace:
            continue
        if ttype in Token.Name:
            tokens.append("ID")
        elif ttype in Token.Literal.Number:
            tokens.append("NUM")
        elif ttype in Token.Literal.String:
            tokens.append("STR")
        else:
            tokens.append(val)
    return tokens


def compute_syntax_similarity(tokens1: list[str], tokens2: list[str]) -> float:
    """Tinh do tuong dong cu phap theo cong thuc BigCloneBench (Svajlenko & Roy)."""
    if not tokens1 or not tokens2:
        return 0.0
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    intersection = sum(min(c1[t], c2[t]) for t in c1 if t in c2)
    s1 = sum(c1.values())
    s2 = sum(c2.values())
    ratio1 = intersection / s1 if s1 > 0 else 0.0
    ratio2 = intersection / s2 if s2 > 0 else 0.0
    return min(ratio1, ratio2)


def main():
    raw_dir = Path("data/raw/coder")
    out_dir = Path("data/processed/coder_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    problem_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and (d / "code").exists()]
    logger.info(f"Tim thay {len(problem_dirs)} thu muc bai tap: {[d.name for d in problem_dirs]}")

    # 1. Load tat ca submissions
    submissions_by_problem: dict[str, list[dict]] = {}
    total_valid = 0

    for pdir in problem_dirs:
        pid = pdir.name
        code_dir = pdir / "code"
        meta_file = pdir / "metadata.jsonl"

        meta_map = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        meta_map[item.get("submission_id")] = item

        submissions = []
        for cfile in code_dir.glob("*.*"):
            try:
                raw_code = cfile.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            cleaned = clean_code(raw_code)
            tokens = tokenize_normalized(cleaned)

            if len(cleaned.split("\n")) < 5 or len(tokens) < 15:
                continue  # Bo qua code qua ngan / khong hop le

            sub_id = cfile.stem.split("_")[0]
            user = meta_map.get(sub_id, {}).get("user", cfile.stem.split("_", 1)[-1])

            submissions.append({
                "submission_id": sub_id,
                "problem_id": pid,
                "user": user,
                "file_path": str(cfile),
                "code": cleaned,
                "tokens": tokens,
            })

        submissions_by_problem[pid] = submissions
        total_valid += len(submissions)
        logger.info(f"  Bai {pid}: {len(submissions)} bai nop hop le")

    logger.info(f"Tong so bai nop hop le tren toan bo he thong: {total_valid}")

    # 2. Sinh cac cap Trùng lặp (Clone - Cung bai toan) va phan tang
    positive_candidates = []
    for pid, subs in submissions_by_problem.items():
        n = len(subs)
        # Tao cac cap khac nhau trong cung 1 bai
        for i in range(n):
            for j in range(i + 1, min(n, i + 30)):  # gioi han de tranh bùng nổ tổ hợp
                s1 = subs[i]
                s2 = subs[j]
                if s1["user"] == s2["user"]:
                    continue  # Tranh cap do cung 1 nguoi nop
                sim = compute_syntax_similarity(s1["tokens"], s2["tokens"])
                
                if sim >= 0.70:
                    ctype = "Type-1/2/ST3"
                elif sim >= 0.50:
                    ctype = "MT3"
                else:
                    ctype = "Type-4"

                positive_candidates.append({
                    "sub1": s1["submission_id"],
                    "sub2": s2["submission_id"],
                    "prob1": pid,
                    "prob2": pid,
                    "user1": s1["user"],
                    "user2": s2["user"],
                    "func1": s1["code"],
                    "func2": s2["code"],
                    "label": 1,
                    "clone_type": ctype,
                    "syntax_similarity": round(sim, 4),
                })

    logger.info(f"Tong so cap Positive ung vien: {len(positive_candidates)}")
    t12_cand = [c for c in positive_candidates if c["clone_type"] == "Type-1/2/ST3"]
    mt3_cand = [c for c in positive_candidates if c["clone_type"] == "MT3"]
    t4_cand = [c for c in positive_candidates if c["clone_type"] == "Type-4"]

    logger.info(f"  -> Type-1/2/ST3 (Sim >= 70%): {len(t12_cand)} cap")
    logger.info(f"  -> MT3 (50% <= Sim < 70%): {len(mt3_cand)} cap")
    logger.info(f"  -> Type-4 (Sim < 50%): {len(t4_cand)} cap")

    # Lay mau dai dien Positive:
    # 50 cap Type-1/2/ST3 (hoac toi da co the) + 30 cap MT3 + 120 cap Type-4 => 200 cap Positive
    random.shuffle(t12_cand)
    random.shuffle(mt3_cand)
    random.shuffle(t4_cand)

    n_t12 = min(len(t12_cand), 50)
    n_mt3 = min(len(mt3_cand), 30)
    n_t4 = 200 - n_t12 - n_mt3

    selected_positives = t12_cand[:n_t12] + mt3_cand[:n_mt3] + t4_cand[:n_t4]
    logger.info(f"Da chon {len(selected_positives)} cap Positive ({n_t12} T1/2/ST3, {n_mt3} MT3, {len(selected_positives)-n_t12-n_mt3} Type-4)")

    # 3. Sinh cac cap Khong trung lap (Negative - Khac bai toan)
    negative_candidates = []
    pids = list(submissions_by_problem.keys())
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            p1, p2 = pids[i], pids[j]
            subs1 = submissions_by_problem[p1]
            subs2 = submissions_by_problem[p2]
            for s1 in subs1[:10]:
                for s2 in subs2[:10]:
                    sim = compute_syntax_similarity(s1["tokens"], s2["tokens"])
                    negative_candidates.append({
                        "sub1": s1["submission_id"],
                        "sub2": s2["submission_id"],
                        "prob1": p1,
                        "prob2": p2,
                        "user1": s1["user"],
                        "user2": s2["user"],
                        "func1": s1["code"],
                        "func2": s2["code"],
                        "label": 0,
                        "clone_type": "Non-clone",
                        "syntax_similarity": round(sim, 4),
                    })

    random.shuffle(negative_candidates)
    selected_negatives = negative_candidates[:len(selected_positives)]
    logger.info(f"Da chon {len(selected_negatives)} cap Negative (Non-clone)")

    # 4. Tron va luu dataset
    all_pairs = selected_positives + selected_negatives
    random.shuffle(all_pairs)
    for idx, pair in enumerate(all_pairs):
        pair["id"] = idx + 1

    out_file = out_dir / "coder_test_pairs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"HOAN TAT: Da luu {len(all_pairs)} cap kiem thu vao {out_file}")

    # Luu thong ke tom tat
    summary = {
        "total_pairs": len(all_pairs),
        "total_positive": len(selected_positives),
        "total_negative": len(selected_negatives),
        "clone_type_breakdown": {
            "Type-1/2/ST3": n_t12,
            "MT3": n_mt3,
            "Type-4": len(selected_positives) - n_t12 - n_mt3,
            "Non-clone": len(selected_negatives),
        },
        "problems": [p.name for p in problem_dirs],
        "created_at": "2026-10-31",
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Tom tat tap kiem thu: {summary}")


if __name__ == "__main__":
    main()
