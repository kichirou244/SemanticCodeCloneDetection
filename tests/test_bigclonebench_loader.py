"""Unit tests cho BigCloneBenchLoader (Java pair loader)."""

import pytest
from pathlib import Path

from src.data.loaders.base import ClonePairSample
from src.data.loaders.bigclonebench import BigCloneBenchLoader


@pytest.fixture
def sample_pairs() -> list[ClonePairSample]:
    return [
        ClonePairSample(
            pair_id="pair-001",
            id1="101",
            id2="102",
            func1="public int add(int a, int b) {\n    return a + b;\n}",
            func2="public int sum(int x, int y) {\n    return x + y;\n}",
            label=1,
            language="java",
            split="train",
        ),
        ClonePairSample(
            pair_id="pair-002",
            id1="103",
            id2="104",
            func1="public void greet() { System.out.println(\"Hi\"); }",
            func2="public void exit() { System.exit(0); }",
            label=0,
            language="java",
            split="train",
        ),
        ClonePairSample(
            pair_id="pair-003",
            id1="105",
            id2="106",
            func1="public boolean isEven(int n) { return n % 2 == 0; }",
            func2="public boolean check(int n) { return (n & 1) == 0; }",
            label=1,
            language="java",
            split="train",
        ),
        ClonePairSample(
            pair_id="pair-004",
            id1="107",
            id2="108",
            func1="public int multiply(int a, int b) { return a * b; }",
            func2="public int divide(int a, int b) { return a / b; }",
            label=0,
            language="java",
            split="train",
        ),
    ]


def test_normalize_code():
    loader = BigCloneBenchLoader()
    raw = "  public void test() {\r\n    int a = 1;\r\n}\r\n  "
    expected = "public void test() {\n    int a = 1;\n}"
    assert loader.normalize_code(raw) == expected
    assert loader.normalize_code("") == ""


def test_coerce_label():
    loader = BigCloneBenchLoader()
    assert loader.coerce_label(True) == 1
    assert loader.coerce_label(False) == 0
    assert loader.coerce_label(1) == 1
    assert loader.coerce_label(0) == 0
    assert loader.coerce_label("true") == 1
    assert loader.coerce_label("False") == 0
    assert loader.coerce_label("1") == 1
    assert loader.coerce_label("0") == 0

    with pytest.raises(ValueError):
        loader.coerce_label("unknown")


def test_parse_row():
    loader = BigCloneBenchLoader()
    raw_row = {
        "id": 12345,
        "id1": 678,
        "id2": 890,
        "func1": "public void a() {}\r\n",
        "func2": "public void b() {}\r\n",
        "label": True,
        "clone_type": "Type-2",
    }
    sample = loader.parse_row(raw_row, split="validation")
    assert sample.pair_id == "12345"
    assert sample.id1 == "678"
    assert sample.id2 == "890"
    assert sample.func1 == "public void a() {}"
    assert sample.func2 == "public void b() {}"
    assert sample.label == 1
    assert sample.split == "validation"
    assert sample.language == "java"
    assert sample.metadata == {"clone_type": "Type-2"}


def test_sample_subset(sample_pairs):
    loader = BigCloneBenchLoader()

    # Subset 2 mẫu cân bằng 50:50
    subset = loader.sample_subset(sample_pairs, n_samples=2, seed=42, balance=True)
    assert len(subset) == 2
    labels = [s.label for s in subset]
    assert 1 in labels and 0 in labels

    # Subset với seed cố định phải tái lập 100%
    sub1 = loader.sample_subset(sample_pairs, n_samples=3, seed=123, balance=True)
    sub2 = loader.sample_subset(sample_pairs, n_samples=3, seed=123, balance=True)
    assert [s.pair_id for s in sub1] == [s.pair_id for s in sub2]


def test_save_load_jsonl(sample_pairs, tmp_path: Path):
    file_path = tmp_path / "test_pairs.jsonl"
    BigCloneBenchLoader.save_jsonl(sample_pairs, file_path)
    assert file_path.exists()

    loaded = BigCloneBenchLoader.load_jsonl(file_path)
    assert len(loaded) == len(sample_pairs)
    assert loaded[0].pair_id == sample_pairs[0].pair_id
    assert loaded[0].func1 == sample_pairs[0].func1
    assert loaded[0].label == sample_pairs[0].label


def test_save_load_parquet(sample_pairs, tmp_path: Path):
    file_path = tmp_path / "test_pairs.parquet"
    BigCloneBenchLoader.save_parquet(sample_pairs, file_path)
    assert file_path.exists()

    loaded = BigCloneBenchLoader.load_parquet(file_path)
    assert len(loaded) == len(sample_pairs)
    assert loaded[0].pair_id == sample_pairs[0].pair_id
    assert loaded[0].label == sample_pairs[0].label
    assert loaded[1].func2 == sample_pairs[1].func2


def test_describe_schema():
    loader = BigCloneBenchLoader()
    schema = loader.describe_schema()
    assert schema["hf_id"] == "google/code_x_glue_cc_clone_detection_big_clone_bench"
    assert schema["language"] == "java"
    assert "splits" in schema
    assert "train" in schema["splits"]
    assert "expected_fields" in schema
    assert "label" in schema["expected_fields"]
