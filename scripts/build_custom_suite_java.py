"""Sinh tap hop pairs.jsonl cho data/custom_suite_java/ voi day du cac loai:
- Type-1: rename, format, comments (label 1)
- Type-2: data type changes, wrappers (label 1)
- Type-3: loop changes, control-flow additions (label 1)
- Type-4: semantic clones with completely different algorithms (label 1)
- Non-clone kho: cung skeleton / cung bai nhung khac nghiep vu (label 0)
- Non-clone cross: khac bai toan hoan toan (label 0)
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "custom_suite_java"

PROBLEMS = [
    {
        "id": "P1",
        "name": "BubbleSort vs Sorters",
        "base": "p1_base_bubblesort.java",
        "v_t1": "p1_v_rename.java",
        "v_t2": "p1_v_types.java",
        "v_t3": "p1_v_while.java",
        "v_t4": "p1_v_quicksort.java",
        "n_hard": "p1_n_reverse.java",
        "t3_note": "vong lap for -> while + flag boolean swapped",
        "t4_note": "BubbleSort vs QuickSort phan hoach de quy (cung sap xep tang dan)",
        "nhard_note": "cung 2 vong lap duyet mang swap, nhung dao nguoc mang",
    },
    {
        "id": "P2",
        "name": "GCD / Arithmetic",
        "base": "p2_base_gcd.java",
        "v_t1": "p2_v_rename.java",
        "v_t2": "p2_v_types.java",
        "v_t3": "p2_v_recursion.java",
        "v_t4": "p2_v_binary_gcd.java",
        "n_hard": "p2_n_digitsum.java",
        "t3_note": "vong lap while -> de quy duoi gcd(b, a % b)",
        "t4_note": "Euclid GCD vs Stein Binary GCD dich bit (cung tim UCLN)",
        "nhard_note": "cung while % va /, UCLN vs tong chu so",
    },
    {
        "id": "P3",
        "name": "Graph Reachability",
        "base": "p3_base_bfs.java",
        "v_t1": "p3_v_rename.java",
        "v_t2": "p3_v_collections.java",
        "v_t3": "p3_v_levels.java",
        "v_t4": "p3_v_dfs.java",
        "n_hard": "p3_n_bipartite.java",
        "t3_note": "BFS mo rong theo tang dem khoang cach ngan nhat",
        "t4_note": "BFS vs DFS de quy (cung kiem tra tinh lien thong duong di)",
        "nhard_note": "cung BFS tren ma tran ke, nhung to mau 2 mau kiem tra do thi phan doi",
    },
    {
        "id": "P4",
        "name": "String Palindrome",
        "base": "p4_base_palindrome.java",
        "v_t1": "p4_v_rename.java",
        "v_t2": "p4_v_types.java",
        "v_t3": "p4_v_clean.java",
        "v_t4": "p4_v_reverse.java",
        "n_hard": "p4_n_reverse_string.java",
        "t3_note": "2 con tro bo qua ky tu trang va khong phan biet hoa thuong",
        "t4_note": "2 con tro so sanh vs StringBuilder.reverse().equals()",
        "nhard_note": "cung 2 con tro swap ky tu, nhung tao chuoi dao nguoc",
    },
    {
        "id": "P5",
        "name": "OOP Linked List Data Structure",
        "base": "p5_base_list.java",
        "v_t1": "p5_v_rename.java",
        "v_t2": "p5_v_types.java",
        "v_t3": "p5_v_while.java",
        "v_t4": "p5_v_dynamic_array.java",
        "n_hard": "p5_n_stack.java",
        "t3_note": "dung dummy sentinel node thay vi head don le",
        "t4_note": "SinglyLinkedList vs DynamicArrayList (cung cung cap add/contains/sum)",
        "nhard_note": "cung struct Node con tro, nhung cai dat LIFO Stack (push/pop/peek)",
    },
]


def build_pairs() -> list[dict]:
    pairs = []
    seen = set()

    def add_pair(a: str, b: str, label: int, category: str, note: str, pair_id: str):
        key = tuple(sorted([a, b]))
        if key in seen:
            return
        seen.add(key)
        # Verify both files exist
        assert (DATA_DIR / a).exists(), f"File missing: {a}"
        assert (DATA_DIR / b).exists(), f"File missing: {b}"
        pairs.append({
            "pair_id": pair_id,
            "a": a,
            "b": b,
            "label": label,
            "category": category,
            "note": note,
        })

    # 1. CLONES
    for prob in PROBLEMS:
        pid = prob["id"]
        # Type-1
        add_pair(prob["base"], prob["v_t1"], 1, "clone_type1", f"{pid}: doi ten bien/comment/format", f"{pid}_t1_rename")
        # Type-2
        add_pair(prob["base"], prob["v_t2"], 1, "clone_type2", f"{pid}: doi kieu du lieu/wrapper", f"{pid}_t2_types")
        # Type-3
        add_pair(prob["base"], prob["v_t3"], 1, "clone_type3", f"{pid}: {prob['t3_note']}", f"{pid}_t3_control")
        # Type-4 Semantic Clone
        add_pair(prob["base"], prob["v_t4"], 1, "clone_type4", f"{pid}: {prob['t4_note']}", f"{pid}_t4_semantic")
        # Inter-variant clones
        add_pair(prob["v_t1"], prob["v_t2"], 1, "clone_type2", f"{pid}: t1 vs t2", f"{pid}_t1_t2")
        add_pair(prob["v_t1"], prob["v_t3"], 1, "clone_type3", f"{pid}: t1 vs t3", f"{pid}_t1_t3")
        add_pair(prob["v_t3"], prob["v_t4"], 1, "clone_type4", f"{pid}: t3 vs t4 (semantic)", f"{pid}_t3_t4")

    # 2. NON-CLONES KHÓ (CÙNG BÀI TOÁN / SKELETON GẦN GIỐNG NHƯNG KHÁC NGỮ NGHĨA)
    for prob in PROBLEMS:
        pid = prob["id"]
        add_pair(prob["base"], prob["n_hard"], 0, "nonclone_kho", f"{pid}: {prob['nhard_note']}", f"{pid}_nonclone_base")
        add_pair(prob["v_t1"], prob["n_hard"], 0, "nonclone_kho", f"{pid}: t1 vs n_hard", f"{pid}_nonclone_t1")
        add_pair(prob["v_t2"], prob["n_hard"], 0, "nonclone_kho", f"{pid}: t2 vs n_hard", f"{pid}_nonclone_t2")
        add_pair(prob["v_t3"], prob["n_hard"], 0, "nonclone_kho", f"{pid}: t3 vs n_hard", f"{pid}_nonclone_t3")
        add_pair(prob["v_t4"], prob["n_hard"], 0, "nonclone_kho", f"{pid}: t4 vs n_hard", f"{pid}_nonclone_t4")

    # 3. NON-CLONES CROSS-PROBLEM (KHÁC BÀI TOÁN HOÀN TOÀN)
    for i in range(len(PROBLEMS)):
        for j in range(i + 1, len(PROBLEMS)):
            p_a = PROBLEMS[i]
            p_b = PROBLEMS[j]
            # base vs base
            add_pair(p_a["base"], p_b["base"], 0, "nonclone_cross", f"Cross {p_a['id']} vs {p_b['id']}", f"X_{p_a['id']}{p_b['id']}_base")
            # t4 vs t4 (khác loại thuật toán phức tạp)
            add_pair(p_a["v_t4"], p_b["v_t4"], 0, "nonclone_cross", f"Cross {p_a['id']} vs {p_b['id']} (advanced)", f"X_{p_a['id']}{p_b['id']}_t4")

    return pairs


def main():
    pairs = build_pairs()
    out_file = DATA_DIR / "pairs.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    clones = [p for p in pairs if p["label"] == 1]
    nonclones = [p for p in pairs if p["label"] == 0]
    t1 = [p for p in pairs if p["category"] == "clone_type1"]
    t2 = [p for p in pairs if p["category"] == "clone_type2"]
    t3 = [p for p in pairs if p["category"] == "clone_type3"]
    t4 = [p for p in pairs if p["category"] == "clone_type4"]
    nkho = [p for p in pairs if p["category"] == "nonclone_kho"]
    ncross = [p for p in pairs if p["category"] == "nonclone_cross"]

    print(f"Tổng số cặp đã sinh: {len(pairs)}")
    print(f"  - Clones (label 1): {len(clones)}")
    print(f"    * Type-1: {len(t1)}")
    print(f"    * Type-2: {len(t2)}")
    print(f"    * Type-3: {len(t3)}")
    print(f"    * Type-4 (Semantic): {len(t4)}")
    print(f"  - Non-clones (label 0): {len(nonclones)}")
    print(f"    * Non-clone khó (cùng skeleton): {len(nkho)}")
    print(f"    * Non-clone cross-problem: {len(ncross)}")


if __name__ == "__main__":
    main()
