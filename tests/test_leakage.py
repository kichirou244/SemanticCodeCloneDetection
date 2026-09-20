"""Test leakage checks phát hiện được overlap khi cố tình gây rò rỉ."""

from src.data.leakage.checks import run_leakage_audit
from src.data.loaders.base import CodeSample


def test_detects_overlap():
    a = CodeSample("a", "int x(){return 1;}", "p1", "cpp")
    b = CodeSample("b", "int x(){return 1;}", "p1", "cpp")
    report = run_leakage_audit({"train": [a], "test": [b]})
    assert report["passed"] is False
