"""Test metrics classification/retrieval tính đúng trên ví dụ nhỏ."""

from src.evaluation.metrics import classification_metrics, mrr, recall_at_k


def test_classification_perfect():
    m = classification_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0


def test_retrieval():
    assert mrr([[0, 1], [1]]) == (0.5 + 1.0) / 2
    assert recall_at_k([[0, 1], [0, 0]], k=2) == 0.5
