"""
Script: generate_hard_negatives.py
Mục tiêu: Hiện thực hóa giải pháp Khai phá mẫu âm tính khó (Hard Negative Mining)
cho bài toán Phát hiện Trùng lặp Mã nguồn Ngữ nghĩa:
1. Quét corpus tìm các cặp có độ tương đồng cú pháp cao (Jaccard > 0.5) nhưng khác chức năng.
2. Sinh mẫu đối kháng qua đột biến toán tử lõi (Operator Mutation: Max <-> Min, Sum <-> Count, ...).
3. Đóng gói tập dữ liệu Hard Negatives phục vụ huấn luyện triệt tiêu cạm bẫy khung sườn.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def tokenize_code(code: str) -> set[str]:
    """Tách từ mã nguồn bỏ khoảng trắng và ký tự đặc biệt."""
    tokens = re.findall(r'[a-zA-Z_]\w*|\d+|[+\-*/%<>=!&|^~]+', code)
    return set(tokens)

def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def mutate_to_hard_negative(code: str) -> Tuple[str, str]:
    """Sinh đột biến ngữ nghĩa đối kháng (Semantic Mutation).
    Giữ nguyên 95% khung sườn nhưng đổi hoàn toàn logic tính toán.
    """
    mutated = code
    desc = "none"
    
    # 1. Đổi toán tử so sánh (Max -> Min hoặc ngược lại)
    if " > " in mutated:
        mutated = mutated.replace(" > ", " < ", 1)
        desc = "swap_greater_to_lesser"
    elif " < " in mutated:
        mutated = mutated.replace(" < ", " > ", 1)
        desc = "swap_lesser_to_greater"
    # 2. Đổi toán tử tích lũy (Sum -> Count hoặc Product)
    elif "sum +=" in mutated or "sum = sum +" in mutated:
        mutated = mutated.replace("sum +=", "count +=", 1)
        mutated = mutated.replace("sum = sum +", "count = count +", 1)
        desc = "swap_sum_to_count"
    elif "+=" in mutated:
        mutated = mutated.replace("+=", "*=", 1)
        desc = "swap_add_to_mult"
        
    return mutated, desc

def mine_hard_negatives_from_suite(suite_dir: Path, ext: str) -> List[Dict]:
    """Khai phá các cặp khó từ mã nguồn có sẵn."""
    files = {p.name: p.read_text(encoding="utf-8") for p in suite_dir.glob(f"*.{ext}")}
    token_cache = {name: tokenize_code(code) for name, code in files.items()}
    
    manifest_path = suite_dir / "pairs.jsonl"
    pairs = [json.loads(ln) for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    
    hard_negatives = []
    
    # 1. Lọc các cặp non-clone nhưng có Jaccard tương đồng cao
    for p in pairs:
        if p["label"] == 0:
            sim = jaccard_similarity(token_cache[p["a"]], token_cache[p["b"]])
            if sim >= 0.50:
                hard_negatives.append({
                    "type": "mined_surface_similarity",
                    "file_a": p["a"],
                    "file_b": p["b"],
                    "code_a": files[p["a"]],
                    "code_b": files[p["b"]],
                    "jaccard_sim": round(sim, 4),
                    "label": 0
                })
                
    # 2. Sinh các mẫu đối kháng đột biến
    for fname, code in list(files.items())[:15]:
        mut_code, desc = mutate_to_hard_negative(code)
        if desc != "none":
            hard_negatives.append({
                "type": f"adversarial_mutation_{desc}",
                "file_a": fname,
                "file_b": f"mutated_{fname}",
                "code_a": code,
                "code_b": mut_code,
                "jaccard_sim": round(jaccard_similarity(tokenize_code(code), tokenize_code(mut_code)), 4),
                "label": 0
            })
            
    return hard_negatives

if __name__ == "__main__":
    java_hn = mine_hard_negatives_from_suite(ROOT / "data" / "custom_suite_java", "java")
    cpp_hn = mine_hard_negatives_from_suite(ROOT / "data" / "custom_suite", "cpp")
    
    print(f"Số lượng Hard Negatives trích xuất thành công:")
    print(f"  - Java: {len(java_hn)} cặp (Jaccard trung bình: {np.mean([x['jaccard_sim'] for x in java_hn]):.3f})")
    print(f"  - C/C++: {len(cpp_hn)} cặp (Jaccard trung bình: {np.mean([x['jaccard_sim'] for x in cpp_hn]):.3f})")
    
    out_file = ROOT / "workspace" / "hard_negatives_mined.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"java": java_hn, "cpp": cpp_hn}, f, indent=2)
    print(f"Đã lưu tập Hard Negatives vào: {out_file}")
