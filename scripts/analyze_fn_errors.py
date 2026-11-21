"""Phân tích thống kê chi tiết các ca lỗi (Error Analysis) trên tập kiểm thử 5.000 cặp BigCloneBench,
đặc biệt tập trung sâu vào các trường hợp False Negative (FN).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def extract_method_name(code: str) -> str:
    # heuristic đơn giản tìm tên phương thức Java
    lines = [l.strip() for l in code.split("\n") if l.strip() and not l.strip().startswith("//")]
    pattern = r"(?:public|protected|private|static|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\("
    for line in lines[:5]:
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    return "unknown"

def tokenize_code(code: str) -> set[str]:
    # tách từ/định danh
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code)
    return set(t.lower() for t in tokens if len(t) > 1)

def compute_jaccard(c1: str, c2: str) -> float:
    s1 = tokenize_code(c1)
    s2 = tokenize_code(c2)
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def categorize_code(code: str) -> str:
    c = code.lower()
    if any(k in c for k in ["messagedigest", "md5", "sha", "cipher", "digest", "encrypt", "aes"]):
        return "Cryptography / Hashing"
    if any(k in c for k in ["inputstream", "outputstream", "filechannel", "bufferedreader", "filewriter", "copy", "fileinputstream"]):
        return "File I/O & Streams"
    if any(k in c for k in ["zipentry", "zipinputstream", "zipoutputstream", "gzip", "decompress"]):
        return "Compression / Archive"
    if any(k in c for k in ["connection", "preparedstatement", "resultset", "executequery", "sql"]):
        return "Database / SQL"
    if any(k in c for k in ["httpurlconnection", "socket", "server_socket", "httpclient", "getresponsecode"]):
        return "Network & HTTP"
    if any(k in c for k in ["documentbuilder", "saxparser", "xmlreader", "jsonobject", "parsexml", "json"]):
        return "XML / JSON Parsing"
    if any(k in c for k in ["jbutton", "jpanel", "jframe", "actionevent", "graphics"]):
        return "GUI / Swing"
    if any(k in c for k in ["thread", "runnable", "synchronized", "executor"]):
        return "Concurrency / Threading"
    return "Other Utilities / Algorithmic"

def main():
    print(">>> Nạp dữ liệu và xác suất kiểm thử...")
    pairs_file = ROOT / "results" / "bcb_5k_test_pairs.jsonl"
    probs_m2_file = ROOT / "results" / "m2_dr_5k_probs.npy"
    probs_m4_file = ROOT / "results" / "m4_ep_5k_probs.npy"

    pairs = []
    with open(pairs_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    y_true = np.array([p["label"] for p in pairs])
    probs_m2 = np.load(probs_m2_file)
    probs_m4 = np.load(probs_m4_file) if probs_m4_file.exists() else None

    th_m2 = 0.52
    th_m4 = 0.30

    preds_m2 = (probs_m2 >= th_m2).astype(int)

    tp_idx = np.where((y_true == 1) & (preds_m2 == 1))[0]
    fn_idx = np.where((y_true == 1) & (preds_m2 == 0))[0]
    fp_idx = np.where((y_true == 0) & (preds_m2 == 1))[0]
    tn_idx = np.where((y_true == 0) & (preds_m2 == 0))[0]

    n_total = len(pairs)
    print(f"Tổng mẫu: {n_total} (Clone: {sum(y_true==1)}, Non-clone: {sum(y_true==0)})")
    print(f"Ma trận nhầm lẫn M2 (GraphCodeBERT, th={th_m2}):")
    print(f"  TP: {len(tp_idx)} | TN: {len(tn_idx)} | FP: {len(fp_idx)} | FN: {len(fn_idx)}")
    print(f"  Tổng số ca sai: {len(fp_idx) + len(fn_idx)} ({((len(fp_idx) + len(fn_idx))/n_total)*100:.2f}%)")
    print(f"  Tỷ lệ FN trên tổng lỗi: {len(fn_idx) / (len(fp_idx) + len(fn_idx))*100:.2f}%\n")

    # =========================================================================
    # 1. Phân phối xác suất của FN
    # =========================================================================
    fn_probs = probs_m2[fn_idx]
    bins = [0.0, 0.05, 0.15, 0.30, 0.45, 0.52]
    counts, _ = np.histogram(fn_probs, bins=bins)
    print("--- 1. PHÂN PHỐI XÁC SUẤT CỦA CÁC CA FN (GraphCodeBERT) ---")
    for i in range(len(counts)):
        p_range = f"[{bins[i]:.2f} - {bins[i+1]:.2f})"
        pct = (counts[i] / len(fn_idx)) * 100
        print(f"  Khoảng {p_range}: {counts[i]:3d} ca ({pct:5.1f}%)")
    print(f"  Trung bình prob FN: {fn_probs.mean():.4f} | Trung vị: {np.median(fn_probs):.4f} | Max: {fn_probs.max():.4f}\n")

    # =========================================================================
    # 2. Thống kê độ dài code (LOC, Chars, Disparity) giữa TP và FN
    # =========================================================================
    def get_stats(indices):
        locs_1, locs_2, max_locs = [], [], []
        len_1, len_2, max_lens = [], [], []
        ratios = []
        jaccards = []
        same_names = []
        categories = []

        for idx in indices:
            p = pairs[idx]
            c1, c2 = p["func1"], p["func2"]
            l1, l2 = len(c1.split("\n")), len(c2.split("\n"))
            ch1, ch2 = len(c1), len(c2)
            locs_1.append(l1); locs_2.append(l2)
            max_locs.append(max(l1, l2))
            len_1.append(ch1); len_2.append(ch2)
            max_lens.append(max(ch1, ch2))
            ratios.append(max(l1, l2) / max(1, min(l1, l2)))
            jaccards.append(compute_jaccard(c1, c2))

            m1, m2 = extract_method_name(c1), extract_method_name(c2)
            same_names.append(m1 == m2 and m1 != "unknown")
            cat1 = categorize_code(c1)
            cat2 = categorize_code(c2)
            categories.append(cat1 if cat1 == cat2 else f"{cat1} -> {cat2}")

        return {
            "mean_loc": np.mean(max_locs),
            "median_loc": np.median(max_locs),
            "mean_chars": np.mean(max_lens),
            "median_chars": np.median(max_lens),
            "pct_long_chars": np.mean(np.array(max_lens) > 2000) * 100, # xấp xỉ > 512 tokens
            "mean_ratio": np.mean(ratios),
            "median_ratio": np.median(ratios),
            "mean_jaccard": np.mean(jaccards),
            "median_jaccard": np.median(jaccards),
            "same_name_pct": np.mean(same_names) * 100,
            "categories": Counter(categories),
        }

    tp_stats = get_stats(tp_idx)
    fn_stats = get_stats(fn_idx)
    fp_stats = get_stats(fp_idx)

    print("--- 2. SO SÁNH ĐẶC TRƯNG HỌC THUẬT: ĐÚNG (TP) VS SAI BỎ SÓT (FN) ---")
    print(f"{'Chỉ số':<35} | {'True Positive (TP)':<20} | {'False Negative (FN)':<20} | {'False Positive (FP)':<20}")
    print("-" * 105)
    print(f"{'Độ dài dòng tối đa (Mean LOC)':<35} | {tp_stats['mean_loc']:<20.1f} | {fn_stats['mean_loc']:<20.1f} | {fp_stats['mean_loc']:<20.1f}")
    print(f"{'Độ dài dòng tối đa (Median LOC)':<35} | {tp_stats['median_loc']:<20.1f} | {fn_stats['median_loc']:<20.1f} | {fp_stats['median_loc']:<20.1f}")
    print(f"{'Số ký tự tối đa (Mean Chars)':<35} | {tp_stats['mean_chars']:<20.1f} | {fn_stats['mean_chars']:<20.1f} | {fp_stats['mean_chars']:<20.1f}")
    print(f"{'Số ký tự tối đa (Median Chars)':<35} | {tp_stats['median_chars']:<20.1f} | {fn_stats['median_chars']:<20.1f} | {fp_stats['median_chars']:<20.1f}")
    print(f"{'Tỷ lệ code dài (> 2.000 ký tự / >512 tok)':<35} | {tp_stats['pct_long_chars']:<19.1f}% | {fn_stats['pct_long_chars']:<19.1f}% | {fp_stats['pct_long_chars']:<19.1f}%")
    print(f"{'Tỷ lệ chênh lệch độ dài (Length Ratio)':<35} | {tp_stats['mean_ratio']:<20.2f} | {fn_stats['mean_ratio']:<20.2f} | {fp_stats['mean_ratio']:<20.2f}")
    print(f"{'Độ trùng từ vựng (Jaccard Overlap)':<35} | {tp_stats['mean_jaccard']*100:<19.1f}% | {fn_stats['mean_jaccard']*100:<19.1f}% | {fp_stats['mean_jaccard']*100:<19.1f}%")
    print(f"{'Cùng tên phương thức (Same Name)':<35} | {tp_stats['same_name_pct']:<19.1f}% | {fn_stats['same_name_pct']:<19.1f}% | {fp_stats['same_name_pct']:<19.1f}%\n")

    # =========================================================================
    # 3. Phân loại miền chức năng (Functionality Domain) của FN
    # =========================================================================
    print("--- 3. PHÂN BỐ CÁC CA FN THEO MIỀN CHỨC NĂNG NGHIỆP VỤ ---")
    for cat, cnt in fn_stats["categories"].most_common(10):
        pct = (cnt / len(fn_idx)) * 100
        print(f"  - {cat:<35}: {cnt:3d} ca ({pct:5.1f}%)")
    print()

    # =========================================================================
    # 4. Kiểm tra mô hình CPG-GNN (M4) có cứu được FN của M2 không?
    # =========================================================================
    if probs_m4 is not None:
        preds_m4 = (probs_m4 >= th_m4).astype(int)
        rescued_by_m4 = np.where((preds_m2[fn_idx] == 0) & (preds_m4[fn_idx] == 1))[0]
        both_fn = np.where((preds_m2[fn_idx] == 0) & (preds_m4[fn_idx] == 0))[0]
        print("--- 4. KHẢ NĂNG CỨU LỖI FN CỦA MÔ HÌNH ĐỒ THỊ CPG-GNN (M4) ---")
        print(f"  Số ca FN của M2 được M4 dự đoán đúng (Rescue): {len(rescued_by_m4)} / {len(fn_idx)} ({len(rescued_by_m4)/len(fn_idx)*100:.1f}%)")
        print(f"  Số ca FN khó mà cả M2 và M4 đều sai (Hard FN): {len(both_fn)} / {len(fn_idx)} ({len(both_fn)/len(fn_idx)*100:.1f}%)\n")

    # =========================================================================
    # 5. Chi tiết các ca FN tiêu biểu (Representative FN Examples)
    # =========================================================================
    print("--- 5. CHI TIẾT 5 TRƯỜNG HỢP FN ĐIỂN HÌNH ---")
    # chọn các ca có xác suất cực thấp
    sorted_fn = sorted(fn_idx, key=lambda i: probs_m2[i])
    samples_to_show = sorted_fn[:5]

    out_records = []
    for rank, idx in enumerate(samples_to_show, 1):
        p = pairs[idx]
        c1, c2 = p["func1"], p["func2"]
        m1, m2 = extract_method_name(c1), extract_method_name(c2)
        jaccard = compute_jaccard(c1, c2)
        p_val = probs_m2[idx]
        p_m4_val = probs_m4[idx] if probs_m4 is not None else 0.0

        rec = {
            "rank": rank,
            "pair_id": p.get("id"),
            "id1": p.get("id1"),
            "id2": p.get("id2"),
            "prob_m2": float(p_val),
            "prob_m4": float(p_m4_val),
            "method1": m1,
            "method2": m2,
            "loc1": len(c1.split("\n")),
            "loc2": len(c2.split("\n")),
            "jaccard": float(jaccard),
            "category1": categorize_code(c1),
            "category2": categorize_code(c2),
            "func1_snippet": c1[:300] + ("..." if len(c1) > 300 else ""),
            "func2_snippet": c2[:300] + ("..." if len(c2) > 300 else ""),
        }
        out_records.append(rec)

        print(f"\n[Ví dụ #{rank}] Pair ID: {p.get('id')} (ID1: {p.get('id1')}, ID2: {p.get('id2')})")
        print(f"  M2 Prob: {p_val:.4f} (M4 Prob: {p_m4_val:.4f}) | Method Names: '{m1}' vs '{m2}'")
        print(f"  LOC: {len(c1.split('\n'))} vs {len(c2.split('\n'))} | Jaccard Overlap: {jaccard*100:.1f}%")
        print(f"  Chức năng: {categorize_code(c1)}  <==>  {categorize_code(c2)}")
        print(f"  Func1 preview:\n    " + c1[:180].replace("\n", "\n    "))
        print(f"  Func2 preview:\n    " + c2[:180].replace("\n", "\n    "))

    # Lưu kết quả phân tích JSON
    analysis_out = ROOT / "results" / "fn_error_analysis_detailed.json"
    with open(analysis_out, "w", encoding="utf-8") as f:
        json.dump({
            "model": "M2_Tuning_DR",
            "threshold": th_m2,
            "tp": len(tp_idx), "tn": len(tn_idx), "fp": len(fp_idx), "fn": len(fn_idx),
            "fn_prob_histogram": {f"[{bins[i]:.2f}-{bins[i+1]:.2f})": int(counts[i]) for i in range(len(counts))},
            "tp_stats": {k: v if not isinstance(v, Counter) else dict(v) for k, v in tp_stats.items()},
            "fn_stats": {k: v if not isinstance(v, Counter) else dict(v) for k, v in fn_stats.items()},
            "fp_stats": {k: v if not isinstance(v, Counter) else dict(v) for k, v in fp_stats.items()},
            "top_fn_examples": out_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n>>> Đã lưu kết quả phân tích chi tiết vào: {analysis_out.name}")

if __name__ == "__main__":
    main()
