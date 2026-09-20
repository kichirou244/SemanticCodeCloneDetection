"""Feasibility AST/CPG: python scripts/check_toolchain.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.representations.ast.extract import check_toolchain
from src.representations.cpg.build import cpg_status

print(json.dumps({"ast": check_toolchain(), "cpg": cpg_status()}, indent=2))
