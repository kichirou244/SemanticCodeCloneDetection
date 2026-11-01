#!/usr/bin/env python3
"""Chuan bi bo du lieu benchmark CODER HUSC (C/C++) phan tang theo BigCloneBench."""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import random
import re

from pygments.lexers import CppLexer
from pygments.token import Token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("prepare_coder")

SEED = 42
random.seed(SEED)
lexer = CppLexer()


def clean_code(source: str) -> str:
    cleaned = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    cleaned = re.sub(r"//.*", "", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in cleaned.split("\n") if line.strip())


def tokenize_normalized(source: str) -> list[str]:
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
    if not tokens1 or not tokens2:
        return 0.0
    c1, c2 = Counter(tokens1), Counter(tokens2)
    intersection = sum(min(c1[t], c2[t]) for t in c1 if t in c2)
    s1, s2 = sum(c1.values()), sum(c2.values())
    return min(intersection / s1, intersection / s2) if s1 > 0 and s2 > 0 else 0.0


def main():
    raw_dir = Path("data/raw/coder")
    out_dir = Path("data/processed/coder_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    problem_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and (d / "code").exists()]
    submissions_by_problem: dict[str, list[dict]] = {}

    for pdir in problem_dirs:
        pid = pdir.name
        code_dir = pdir / "code"
        meta_file = pdir / "metadata.jsonl"
        meta_map = {}
        if meta_file.exists():
            with open(meta_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        meta_map[item.get("submission_id")] = item

        submissions = []
        for cfile in code_dir.glob("*.*"):
            try:
                raw = cfile.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            cleaned = clean_code(raw)
            tokens = tokenize_normalized(cleaned)
            if len(cleaned.split("\n")) < 5 or len(tokens) < 15:
                continue

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

    # Sinh cac cap Clone (Positive)
    positive_candidates = []
    for pid, subs in submissions_by_problem.items():
        n = len(subs)
        for i in range(n):
            for j in range(i + 1, min(n, i + 30)):
                s1, s2 = subs[i], subs[j]
                if s1["user"] == s2["user"]:
                    continue
                sim = compute_syntax_similarity(s1["tokens"], s2["tokens"])
                ctype = "Type-1/2/ST3" if sim >= 0.70 else ("MT3" if sim >= 0.50 else "Type-4")
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

    t12 = [c for c in positive_candidates if c["clone_type"] == "Type-1/2/ST3"]
    mt3 = [c for c in positive_candidates if c["clone_type"] == "MT3"]
    t4 = [c for c in positive_candidates if c["clone_type"] == "Type-4"]

    random.shuffle(t12)
    random.shuffle(mt3)
    random.shuffle(t4)

    n_t12 = min(len(t12), 50)
    n_mt3 = min(len(mt3), 30)
    n_t4 = 200 - n_t12 - n_mt3
    selected_positives = t12[:n_t12] + mt3[:n_mt3] + t4[:n_t4]

    # Sinh cac cap Non-clone (Negative)
    negative_candidates = []
    pids = list(submissions_by_problem.keys())
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            s1_list = submissions_by_problem[pids[i]]
            s2_list = submissions_by_problem[pids[j]]
            for s1 in s1_list[:10]:
                for s2 in s2_list[:10]:
                    sim = compute_syntax_similarity(s1["tokens"], s2["tokens"])
                    negative_candidates.append({
                        "sub1": s1["submission_id"],
                        "sub2": s2["submission_id"],
                        "prob1": pids[i],
                        "prob2": pids[j],
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

    all_pairs = selected_positives + selected_negatives
    random.shuffle(all_pairs)
    for idx, pair in enumerate(all_pairs):
        pair["id"] = idx + 1

    out_file = out_dir / "coder_test_pairs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "total_pairs": len(all_pairs),
        "total_positive": len(selected_positives),
        "total_negative": len(selected_negatives),
        "breakdown": {
            "Type-1/2/ST3": n_t12,
            "MT3": n_mt3,
            "Type-4": n_t4,
            "Non-clone": len(selected_negatives),
        },
        "problems": [p.name for p in problem_dirs],
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Da luu %d cap kiem thu vao %s", len(all_pairs), out_file)


if __name__ == "__main__":
    main()
