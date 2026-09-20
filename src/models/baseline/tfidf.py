"""B1 — TF-IDF + Cosine Similarity baseline (lexical, không học sâu)."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.interfaces import CloneDetector, ClonePair


def code_tokenizer(code: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\S", code or "")


class TfidfCloneDetector(CloneDetector):
    def __init__(self, max_features: int = 20000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=code_tokenizer, token_pattern=None, max_features=max_features
        )

    def fit(self, codes: list[str]) -> None:
        self.vectorizer.fit(codes)

    def predict_proba(self, pairs: list[ClonePair]) -> list[float]:
        if not pairs:
            return []
        flat = [p.code_a for p in pairs] + [p.code_b for p in pairs]
        mat = self.vectorizer.transform(flat)
        n = len(pairs)
        va, vb = mat[:n], mat[n:]
        sims = cosine_similarity(va, vb).diagonal()
        return [float(max(0.0, min(1.0, s))) for s in sims]
