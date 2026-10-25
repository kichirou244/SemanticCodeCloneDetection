"""Mo rong bo test `custom_suite` voi cac cap KHONG trung lap (label 0).

Boi canh
--------
Bo test cu co 25 cap clone + 16 cap non-clone, trong do chi 6 cap non-clone
"kho" (cung khung, khac ngu nghia). Thuc te phan lon cap dua vao he thong la
non-clone, va day la lop de bi bo qua khi bo test qua nho. Script nay sinh them
cap non-clone theo hai luat **tat dinh**, khong cham vao cac cap clone da co:

1. `nonclone_kho` — trong cung mot bai, cap file `pX_n_*.cpp` (khac ngu nghia)
   voi **tat ca** file clone cua bai do (base + cac bien the). Vi du
   `p1_n_fact.cpp` vs `p1_v_while.cpp`. Day la cac cap cung khung, khac loi.
2. `nonclone_de` — cap khac bai (khac problem) nhung co cung kieu bien doi
   (while/dowhile/scanf/combo) hoac cung nhom base. Khac problem => chac chan
   khong clone, nhung van co the lam mo hinh nham neu chung chia se khung.

Chay::

    python scripts/build_custom_suite_negatives.py --check   # chi kiem tra
    python scripts/build_custom_suite_negatives.py --write   # ghi vao pairs.jsonl

Script idempotent: chay lai khong them dong trung, khong doi thu tu cap cu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "custom_suite"
PAIRS = DATA_DIR / "pairs.jsonl"

#: Cau truc tung bai. `variants` gom ca cac bien the clone (khong gom file n_).
GROUPS: dict[str, dict] = {
    "P1": {
        "name": "sum 1..n",
        "base": "p1_base_sum.cpp",
        "variants": ["p1_v_while.cpp", "p1_v_dowhile.cpp",
                     "p1_v_scanf.cpp", "p1_v_combo.cpp"],
        "nonclone": "p1_n_fact.cpp",
        "nonclone_short": "fact",
        "nonclone_desc": "tong 1..n vs giai thua",
    },
    "P2": {
        "name": "UCLN",
        "base": "p2_base_gcd.cpp",
        "variants": ["p2_v_for.cpp", "p2_v_dowhile.cpp",
                     "p2_v_scanf.cpp", "p2_v_combo.cpp"],
        "nonclone": "p2_n_digitsum.cpp",
        "nonclone_short": "digitsum",
        "nonclone_desc": "UCLN vs tong chu so",
    },
    "P3": {
        "name": "max mang",
        "base": "p3_base_max.cpp",
        "variants": ["p3_v_while.cpp", "p3_v_dowhile.cpp",
                     "p3_v_scanf.cpp", "p3_v_combo.cpp"],
        "nonclone": "p3_n_sum.cpp",
        "nonclone_short": "sum",
        "nonclone_desc": "max mang vs tong mang",
    },
    "P4": {
        "name": "so nguyen to",
        "base": "p4_base_prime.cpp",
        "variants": ["p4_v_while.cpp", "p4_v_dowhile.cpp",
                     "p4_v_scanf.cpp", "p4_v_combo.cpp"],
        "nonclone": "p4_n_square.cpp",
        "nonclone_short": "square",
        "nonclone_desc": "so nguyen to vs so chinh phuong",
    },
    "P5": {
        "name": "DSLK co ban + OOP",
        "base": "p5_base_list.cpp",
        "variants": ["p5_v_class.cpp", "p5_v_while.cpp",
                     "p5_v_scanf.cpp", "p5_v_combo.cpp"],
        "nonclone": "p5_n_palindrome.cpp",
        "nonclone_short": "palindrome",
        "nonclone_desc": "tong DSLK vs kiem tra doi xung",
    },
    "P6": {
        "name": "DSLK co thu tu",
        "base": "p6_base_list.cpp",
        "variants": ["p6_v_class.cpp", "p6_v_while.cpp",
                     "p6_v_scanf.cpp", "p6_v_combo.cpp"],
        "nonclone": "p6_n_filter.cpp",
        "nonclone_short": "filter",
        "nonclone_desc": "chen co thu tu + xoa trung vs loc nguong + dem le",
    },
}

#: Cap cross-problem theo cung kieu bien doi. (bai A, hau to A, bai B, hau to B)
#: `v_for` cua P2 duoc dung thay `v_while` vi P2 khong co ban while.
TRANSFORM_PAIRS: list[tuple[str, str, str, str, str]] = [
    # (transform, group_a, suffix_a, group_b, suffix_b)
    ("while", "P1", "v_while", "P2", "v_for"),
    ("while", "P3", "v_while", "P4", "v_while"),
    ("while", "P5", "v_while", "P6", "v_while"),
    ("while", "P1", "v_while", "P3", "v_while"),

    ("dowhile", "P1", "v_dowhile", "P2", "v_dowhile"),
    ("dowhile", "P2", "v_dowhile", "P3", "v_dowhile"),
    ("dowhile", "P3", "v_dowhile", "P4", "v_dowhile"),
    ("dowhile", "P1", "v_dowhile", "P4", "v_dowhile"),

    ("scanf", "P1", "v_scanf", "P2", "v_scanf"),
    ("scanf", "P3", "v_scanf", "P4", "v_scanf"),
    ("scanf", "P5", "v_scanf", "P6", "v_scanf"),
    ("scanf", "P1", "v_scanf", "P3", "v_scanf"),

    ("combo", "P1", "v_combo", "P2", "v_combo"),
    ("combo", "P3", "v_combo", "P4", "v_combo"),
    ("combo", "P5", "v_combo", "P6", "v_combo"),
    ("combo", "P2", "v_combo", "P3", "v_combo"),
    # Hai cap bo sung: cung kieu combo nhung khac bai hoan toan.
    ("combo", "P2", "v_combo", "P5", "v_combo"),
    ("combo", "P3", "v_combo", "P6", "v_combo"),
]

#: Cap base-base con thieu (tat ca 15 cap giua 6 bai).
BASE_PAIRS: list[tuple[str, str]] = [
    ("P1", "P6"), ("P2", "P6"), ("P3", "P6"),
    ("P4", "P6"), ("P3", "P5"), ("P4", "P5"),
]


def _read_pairs() -> list[dict]:
    rows = []
    with PAIRS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _suffix_to_word(suffix: str) -> str:
    return suffix.replace("v_", "")


def build_new_rows(existing: list[dict]) -> list[dict]:
    """Sinh cac dong non-clone chua co trong `existing` (giu nguyen thu tu)."""
    existing_ids = {r["pair_id"] for r in existing}
    existing_pairs = {frozenset((r["a"], r["b"])) for r in existing}
    rows: list[dict] = []

    def add(pid: str, a: str, b: str, category: str, note: str) -> None:
        if pid in existing_ids:
            return
        key = frozenset((a, b))
        if key in existing_pairs:
            # Cap da co duoi id khac — khong them trung.
            return
        rows.append({
            "a": a, "b": b, "category": category, "label": 0,
            "note": note, "pair_id": pid,
        })
        existing_ids.add(pid)
        existing_pairs.add(key)

    # 1) Non-clone kho: file n_ cua bai vs moi file clone cua cung bai.
    for gid, g in GROUPS.items():
        clone_files = [g["base"], *g["variants"]]
        for f in clone_files:
            suffix = _suffix_to_word(Path(f).stem.split("_", 2)[-1])
            # p1_v_while -> while; p1_base_sum -> base_sum
            add(
                f"{gid}_nonclone_{g['nonclone_short']}__{suffix}",
                g["nonclone"], f, "nonclone_kho",
                f"{gid}: {g['nonclone_desc']} (bien the {suffix})",
            )

    # 2a) Base-base con thieu.
    for ga, gb in BASE_PAIRS:
        add(
            f"X_{ga}{gb}", GROUPS[ga]["base"], GROUPS[gb]["base"],
            "nonclone_de",
            f"cross: {GROUPS[ga]['name']} vs {GROUPS[gb]['name']} "
            f"(2 bai khac nhau)",
        )

    # 2b) Cung kieu bien doi, khac bai.
    for transform, ga, sa, gb, sb in TRANSFORM_PAIRS:
        a = f"{ga.lower()}_{sa}.cpp"
        b = f"{gb.lower()}_{sb}.cpp"
        add(
            f"X_{ga}{gb}_{transform}", a, b, "nonclone_de",
            f"cross: cung kieu {transform} nhung {GROUPS[ga]['name']} "
            f"vs {GROUPS[gb]['name']} (2 bai khac nhau)",
        )

    return rows


def validate(rows: list[dict]) -> None:
    """Kiem tra file ton tai, khong trung cap, nhan dung, so luong dat muc tieu."""
    files = {p.name for p in DATA_DIR.glob("*.cpp")}
    seen: set[frozenset] = set()
    ids: set[str] = set()
    for r in rows:
        for key in ("a", "b"):
            if r[key] not in files:
                raise SystemExit(f"thieu file {r[key]} (pair {r['pair_id']})")
        if r["a"] == r["b"]:
            raise SystemExit(f"cap tu than: {r['pair_id']}")
        key = frozenset((r["a"], r["b"]))
        if key in seen:
            raise SystemExit(f"cap trung: {r['pair_id']} ({r['a']} vs {r['b']})")
        if r["pair_id"] in ids:
            raise SystemExit(f"pair_id trung: {r['pair_id']}")
        seen.add(key)
        ids.add(r["pair_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description="sinh cap non-clone cho custom_suite")
    ap.add_argument("--write", action="store_true", help="ghi vao pairs.jsonl")
    ap.add_argument("--check", action="store_true", help="chi kiem tra, khong ghi")
    args = ap.parse_args()
    if not args.write and not args.check:
        ap.error("chon --check hoac --write")

    existing = _read_pairs()
    new = build_new_rows(existing)
    all_rows = existing + new
    validate(all_rows)

    n_clone = sum(1 for r in all_rows if int(r["label"]) == 1)
    n_non = sum(1 for r in all_rows if int(r["label"]) == 0)
    by_cat: dict[str, int] = {}
    for r in all_rows:
        if int(r["label"]) == 0:
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1

    print(f"hien co : {len(existing)} cap "
          f"({sum(1 for r in existing if r['label'] == 1)} clone + "
          f"{sum(1 for r in existing if r['label'] == 0)} non-clone)")
    print(f"them moi: {len(new)} cap")
    print(f"sau khi them: {len(all_rows)} cap "
          f"({n_clone} clone + {n_non} non-clone, "
          f"non-clone gap {n_non / max(sum(1 for r in existing if r['label'] == 0), 1):.2f} lan ban cu)")
    print("non-clone theo nhom:", by_cat)
    for r in new:
        print(f"  + {r['pair_id']:<38} {r['a']} vs {r['b']}")

    if args.write:
        with PAIRS.open("w", encoding="utf-8", newline="\n") as fh:
            for r in all_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"da ghi {PAIRS}")
    else:
        print("(--check: khong ghi)")


if __name__ == "__main__":
    main()
