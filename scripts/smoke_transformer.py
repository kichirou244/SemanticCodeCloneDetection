"""Smoke test M1/M2: load model → tokenize → forward → embedding (cần mạng + RAM).

Chạy: python scripts/smoke_transformer.py --method codebert
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.seed import set_global_seed
from src.models.transformer.embedder import TransformerEmbedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["codebert", "graphcodebert"], default="codebert")
    args = ap.parse_args()
    set_global_seed(42)
    emb = TransformerEmbedder(args.method).encode(
        ["int add(int a,int b){return a+b;}", "def add(a,b): return a+b"]
    )
    print(f"{args.method}: embedding shape = {tuple(emb.shape)} — OK")


if __name__ == "__main__":
    main()
