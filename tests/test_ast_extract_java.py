"""Unit tests cho module trích xuất AST đối với ngôn ngữ Java (tree-sitter-java)."""

import numpy as np
import pytest

from src.representations.ast.extract import (
    check_toolchain,
    collate_trees,
    compile_tree,
    depth_of,
    extract_tree,
    get_parser,
    tree_stats,
)


def test_toolchain_java():
    report = check_toolchain()
    assert report["tree_sitter"] is True
    assert "java" in report["grammars"]
    assert report["grammars"]["java"] is True


def test_get_parser_java():
    parser = get_parser("java")
    assert parser is not None
    tree = parser.parse(b"public class Test {}")
    assert tree.root_node.type == "program"


def test_extract_tree_simple_method():
    code = """
    public int calculateSum(int a, int b) {
        int result = a + b;
        return result;
    }
    """
    tree = extract_tree(code, language="java")
    assert "label" in tree
    assert "parent" in tree
    assert "is_leaf" in tree
    assert len(tree["label"]) > 0

    stats = tree_stats(tree)
    assert stats["n_nodes"] > 0
    assert stats["n_stmt"] > 0
    assert stats["n_leaf"] > 0
    assert stats["max_depth"] >= 2


def test_compile_tree_java():
    code = """
    public void demo() {
        int x = 10;
        if (x > 5) {
            System.out.println("Greater");
        } else {
            System.out.println("Smaller");
        }
    }
    """
    tree = extract_tree(code, language="java")
    compiled = compile_tree(tree)

    assert "s_label" in compiled
    assert "s_parent" in compiled
    assert "s_depth" in compiled
    assert "anc_s" in compiled
    assert "anc_c" in compiled
    assert compiled["n_stmt"] > 0
    assert compiled["n_top"] > 0

    # Kiểm tra tính toàn vẹn: s_parent của các con luôn nhỏ hơn chỉ số của nó (DFS order)
    n_stmt = int(compiled["n_stmt"])
    if n_stmt > 1:
        assert np.all(compiled["s_parent"][1:] < np.arange(1, n_stmt))


def test_java_advanced_syntax():
    # Kiểm tra các cú pháp đặc thù của Java: try-with-resources, enhanced-for, lambda, annotations
    code = """
    @Override
    public List<String> processItems(List<String> items) {
        List<String> output = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader("data.txt"))) {
            for (String item : items) {
                if (item != null && !item.isEmpty()) {
                    output.add(item.trim());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            output.forEach(s -> System.out.println(s));
        }
        return output;
    }
    """
    tree = extract_tree(code, language="java")
    compiled = compile_tree(tree)
    assert compiled["n_stmt"] >= 5
    assert len(compiled["anc_s"]) > 0


def test_collate_java_trees():
    code1 = "public int add(int a, int b) { return a + b; }"
    code2 = "public int mul(int a, int b) { int c = a * b; return c; }"

    tree1 = extract_tree(code1, language="java")
    tree2 = extract_tree(code2, language="java")

    c1 = compile_tree(tree1)
    c2 = compile_tree(tree2)

    batch = collate_trees([c1, c2])
    assert batch["batch_size"] == 2
    assert "s_label" in batch
    assert "s_parent" in batch
    assert "s_order" in batch
    assert "top_idx" in batch
    assert batch["top_idx"].shape[0] == 2


def test_real_bcb_samples():
    # Kiểm tra trích xuất trên dữ liệu BigCloneBench thật vừa tải về
    from pathlib import Path
    import pandas as pd

    sample_parquet = Path("data/raw/bigclonebench/bcb_smoke.parquet")
    if not sample_parquet.exists():
        pytest.skip("Chưa có bcb_smoke.parquet")

    df = pd.read_parquet(sample_parquet)
    # Lấy 10 mẫu bất kỳ
    for i in range(min(10, len(df))):
        func1 = df.iloc[i]["func1"]
        func2 = df.iloc[i]["func2"]

        t1 = extract_tree(func1, language="java")
        t2 = extract_tree(func2, language="java")

        c1 = compile_tree(t1)
        c2 = compile_tree(t2)

        assert c1["n_stmt"] > 0
        assert c2["n_stmt"] > 0

        # Ghép cặp kiểm tra collation
        batch = collate_trees([c1, c2])
        assert batch["batch_size"] == 2
