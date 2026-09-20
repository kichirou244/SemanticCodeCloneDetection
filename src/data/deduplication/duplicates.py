"""Phát hiện exact duplicate + normalized duplicate."""

import hashlib
import re


def normalize(code: str) -> str:
    code = re.sub(r"\s+", " ", code or "").strip()
    return code


def fingerprint(code: str, normalized: bool = True) -> str:
    text = normalize(code) if normalized else (code or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicate_stats(codes: list[str]) -> dict:
    exact = {}
    normed = {}
    for c in codes:
        exact[fingerprint(c, False)] = exact.get(fingerprint(c, False), 0) + 1
        normed[fingerprint(c, True)] = normed.get(fingerprint(c, True), 0) + 1
    return {
        "n": len(codes),
        "n_unique_exact": len(exact),
        "n_unique_normalized": len(normed),
        "n_dup_exact": len(codes) - len(exact),
        "n_dup_normalized": len(codes) - len(normed),
    }
