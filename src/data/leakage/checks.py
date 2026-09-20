"""Leakage checks: problem/sample/code/pair overlap xuyên split (Steps 18-19)."""

from src.data.deduplication.duplicates import fingerprint
from src.data.loaders.base import CodeSample


def _pairwise(keys: list[str], sets: dict[str, set]) -> tuple[dict, bool]:
    overlap = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap[f"{keys[i]} x {keys[j]}"] = len(sets[keys[i]] & sets[keys[j]])
    return overlap, all(v == 0 for v in overlap.values())


def check_problem_overlap(splits: dict[str, list[CodeSample]]) -> dict:
    probs = {k: {s.problem_id for s in v} for k, v in splits.items()}
    keys = list(probs)
    overlap = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = probs[keys[i]] & probs[keys[j]]
            overlap[f"{keys[i]} x {keys[j]}"] = len(inter)
    return {"problem_overlap": overlap, "ok": all(v == 0 for v in overlap.values())}


def check_code_overlap(splits: dict[str, list[CodeSample]]) -> dict:
    """Exact = byte-identical sha256(raw). Không collapse whitespace ở đây
    (định nghĩa duy nhất cho normalized nằm ở CodeNormalizer)."""
    fps = {k: {fingerprint(s.code, normalized=False) for s in v} for k, v in splits.items()}
    keys = list(fps)
    overlap = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap[f"{keys[i]} x {keys[j]}"] = len(fps[keys[i]] & fps[keys[j]])
    return {"code_overlap": overlap, "ok": all(v == 0 for v in overlap.values())}


def run_leakage_audit(splits: dict[str, list[CodeSample]]) -> dict:
    p = check_problem_overlap(splits)
    c = check_code_overlap(splits)
    return {"problem": p, "code": c, "passed": p["ok"] and c["ok"]}


def check_sample_overlap(splits: dict[str, list[CodeSample]]) -> dict:
    """Sample ID không được xuất hiện ở 2 split (Step 18)."""
    sets = {k: {s.sample_id for s in v} for k, v in splits.items()}
    overlap, ok = _pairwise(list(sets), sets)
    return {"sample_overlap": overlap, "ok": ok}


def check_normalized_code_overlap(splits: dict[str, list[CodeSample]]) -> dict:
    sets = {
        k: {fingerprint(s.normalized_code or s.code, normalized=False) for s in v}
        for k, v in splits.items()
    }
    overlap, ok = _pairwise(list(sets), sets)
    return {"normalized_code_overlap": overlap, "ok": ok}


def canonical_pair(a: str, b: str) -> str:
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}:::{hi}"


def check_pair_overlap(pair_splits: dict[str, list]) -> dict:
    """Pair (A,B) vô thứ tự không xuất hiện ở 2 split (Step 19)."""
    sets = {}
    for k, pairs in pair_splits.items():
        ids = set()
        for p in pairs:
            a = p["code_a_id"] if isinstance(p, dict) else p.code_a_id
            b = p["code_b_id"] if isinstance(p, dict) else p.code_b_id
            ids.add(canonical_pair(a, b))
        sets[k] = ids
    overlap, ok = _pairwise(list(sets), sets)
    return {"pair_overlap": overlap, "ok": ok}


def run_pair_leakage_audit(
    sample_splits: dict[str, list[CodeSample]], pair_splits: dict[str, list]
) -> dict:
    s = check_sample_overlap(sample_splits)
    n = check_normalized_code_overlap(sample_splits)
    pl = check_pair_overlap(pair_splits)
    base = run_leakage_audit(sample_splits)
    passed = base["passed"] and s["ok"] and n["ok"] and pl["ok"]
    return {
        "problem": base["problem"],
        "code": base["code"],
        "sample": s,
        "normalized_code": n,
        "pair": pl,
        "passed": passed,
    }
