"""Smoke test baseline: fit TF-IDF trên 4 mẫu, kiểm tra shape output."""

from src.core.interfaces import ClonePair
from src.models.baseline.tfidf import TfidfCloneDetector


def test_tfidf_smoke():
    det = TfidfCloneDetector(max_features=100)
    det.fit(["int a(){return 1;}", "def a(): return 1"])
    scores = det.predict_proba([ClonePair("int a(){return 1;}", "def a(): return 1", 1)])
    assert len(scores) == 1 and 0.0 <= scores[0] <= 1.0
