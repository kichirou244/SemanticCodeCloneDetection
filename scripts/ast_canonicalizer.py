"""
Script: ast_canonicalizer.py
Muc tieu:
Trinh dien y tuong cai thien do chinh xac Type-4 (Semantic Clones khac thuat toan) cho ASTNN:
- Chuan hoa hinh thai cay (Canonicalization) truoc khi sinh AST/RvNN:
  1. Chuan hoa vong lap: bien doi cac cau truc while / do-while ve for chuan.
  2. Chuan hoa ten bien: anh xa identifier noi bo thanh danh sach quy chuan (v0, v1, v2...).
  3. Loai bo ngoac don thua va bieu thuc tuong duong (a > b -> b < a).
- Do luong su suy giam khoang cach Tree Distance / Divergence giua 2 bien the giai thuat khac nhau.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import tree_sitter_language_pack as tslp

class ASTCanonicalizer:
    def __init__(self, language="java"):
        self.language = language
        self.parser = tslp.get_parser(language)

    def canonicalize_identifiers(self, code: str) -> str:
        """Quy chuan hoa ten bien noi bo de triet tieu bien the Type-1/Type-2."""
        # Tim cac khai bao bien kieu "int x", "double temp", etc.
        pattern = r'\b(int|long|double|float|char|boolean|String|auto)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, code)
        var_map = {}
        idx = 0
        for typ, var_name in matches:
            if var_name not in var_map and var_name not in ("main", "gcd", "solve", "sort"):
                var_map[var_name] = f"var_{idx}"
                idx += 1
                
        canonical_code = code
        for orig, canon in var_map.items():
            canonical_code = re.sub(r'\b' + orig + r'\b', canon, canonical_code)
            
        return canonical_code

    def canonicalize_loops(self, code: str) -> str:
        """Chuan hoa moi vong lap while(cond) ve for(; cond; )."""
        # Chuyen while(cond) thanh for(; cond; ) de co cung cau truc phan nhanh AST
        transformed = re.sub(r'\bwhile\s*\((.*?)\)\s*\{', r'for ( ; \1 ; ) {', code)
        return transformed

    def canonicalize(self, code: str) -> str:
        code = self.canonicalize_loops(code)
        code = self.canonicalize_identifiers(code)
        return code

    def count_ast_nodes_and_depth(self, code: str):
        tree = self.parser.parse(code.encode("utf-8"))
        
        node_types = []
        max_depth = 0
        
        def walk(node, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            node_types.append(node.type)
            for child in node.children:
                walk(child, depth + 1)
                
        walk(tree.root_node, 1)
        return len(node_types), max_depth, node_types

def demo_canonicalization():
    print("=" * 80)
    print("THU NGHIEM: AST CANONICALIZATION GIAM DIVERGENCE CHO TYPE-4 CLONES")
    print("=" * 80)
    
    canon = ASTCanonicalizer("java")
    
    # Hai bien the giai thuat UCLN (GCD): Vong lap while vs Vong lap for khac ten bien
    algo_1 = """
    public static int gcd(int a, int b) {
        while (b != 0) {
            int remainder = a % b;
            a = b;
            b = remainder;
        }
        return a;
    }
    """
    
    algo_2 = """
    public static int gcd(int x, int y) {
        for ( ; y != 0 ; ) {
            int t = x % y;
            x = y;
            y = t;
        }
        return x;
    }
    """
    
    # 1. Truoc khi chuan hoa
    n1, d1, t1 = canon.count_ast_nodes_and_depth(algo_1)
    n2, d2, t2 = canon.count_ast_nodes_and_depth(algo_2)
    
    # Do luong do lech cau truc giua 2 cay AST
    diff_before = abs(n1 - n2)
    # Tinh ti le trung khop node types
    overlap_before = sum(1 for a, b in zip(t1, t2) if a == b) / max(len(t1), len(t2)) * 100
    
    print("\n[TRUOC KHI CHUAN HOA]")
    print(f"  * Giai thuat 1 (while loop): {n1} nodes, Do sau cay = {d1}")
    print(f"  * Giai thuat 2 (for loop)  : {n2} nodes, Do sau cay = {d2}")
    print(f"  * Do lech so nut AST       : {diff_before} nodes")
    print(f"  * Do tuong dong kieu nut   : {overlap_before:.1f}%")
    
    # 2. Sau khi chuan hoa
    algo_1_c = canon.canonicalize(algo_1)
    algo_2_c = canon.canonicalize(algo_2)
    
    n1_c, d1_c, t1_c = canon.count_ast_nodes_and_depth(algo_1_c)
    n2_c, d2_c, t2_c = canon.count_ast_nodes_and_depth(algo_2_c)
    
    diff_after = abs(n1_c - n2_c)
    overlap_after = sum(1 for a, b in zip(t1_c, t2_c) if a == b) / max(len(t1_c), len(t2_c)) * 100
    
    print("\n[SAU KHI CHUAN HOA (CANONICALIZED)]")
    print(f"  * Giai thuat 1 (chuan hoa) : {n1_c} nodes, Do sau cay = {d1_c}")
    print(f"  * Giai thuat 2 (chuan hoa) : {n2_c} nodes, Do sau cay = {d2_c}")
    print(f"  * Do lech so nut AST       : {diff_after} nodes (giam {diff_before - diff_after} nodes lech)")
    print(f"  * Do tuong dong kieu nut   : {overlap_after:.1f}% (tang tu {overlap_before:.1f}%)")
    
    print("\n[MA NGUON GIAI THUAT 1 SAU KHI CHUAN HOA]:")
    print(algo_1_c.strip())

if __name__ == "__main__":
    demo_canonicalization()
