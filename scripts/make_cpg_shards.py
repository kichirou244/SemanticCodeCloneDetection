"""Dong goi 52k CPG JSON thanh ~52 shard `.jsonl` de Kaggle attach nhanh.

Vi sao: Kaggle TU BUNG moi archive khi tao dataset, nen 52k JSON roi khien
moi lan attach phai tai tung file (cham, treo). File `.jsonl` thuong Kaggle
khong dung toi: download vai chuc file to nhanh hon hang chuc nghin file nho
rat nhieu. Notebook train bung shard ra lai JSON nhu cu — code train khong
doi mot dong.

Dung ca local lan Kaggle:
    python scripts/make_cpg_shards.py --src <zip|thu muc> --out <thu muc shards>
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path


def iter_jsons(src: Path):
    """Yield (file_name, bytes) cho moi function JSON, bo benchmark*.json."""
    if src.is_file() and zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                base = n.rsplit("/", 1)[-1]
                if base.endswith(".json") and "benchmark" not in base:
                    yield base, z.read(n)
    elif src.is_dir():
        for p in sorted(src.glob("*.json")):
            if "benchmark" in p.name:
                continue
            yield p.name, p.read_bytes()
    else:
        raise SystemExit(f"--src phai la zip hoac thu muc: {src}")


def materialize(shard_dirs: list[Path], out_dir: Path) -> int:
    """Bung shards ra lai JSON goc (ten file giu nguyen). Tra ve so file da ghi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for d in shard_dirs:
        for sp in sorted(Path(d).glob("cpg_shard_*.jsonl")):
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    (out_dir / r["file"]).write_text(
                        json.dumps(r["data"], ensure_ascii=False), encoding="utf-8")
                    n += 1
            print(f"  ... xong {sp.name} (tong {n})", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="zip hoac thu muc chua *.json (che do chia shard)")
    ap.add_argument("--out", help="thu muc ghi shard (che do chia) hoac JSON (che do bung)")
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--from-shards", nargs="+",
                    help="thu muc chua shard (che do bung) — can kem --out")
    args = ap.parse_args()

    if args.from_shards:
        if not args.out:
            raise SystemExit("--from-shards can kem --out")
        n = materialize([Path(d) for d in args.from_shards], Path(args.out))
        print(f"[materialize] {n} JSON <- shards")
        return

    if not args.src or not args.out:
        raise SystemExit("can --src + --out (hoac --from-shards + --out)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard_names: list[str] = []
    buf: list[tuple[str, bytes]] = []
    n_files = 0

    def flush() -> None:
        name = f"cpg_shard_{len(shard_names):02d}.jsonl"
        with open(out / name, "w", encoding="utf-8") as f:
            for fname, raw in buf:
                d = json.loads(raw)  # fail-fast neu JSON hong
                f.write(json.dumps({"file": fname, "data": d},
                                   ensure_ascii=False) + "\n")
        shard_names.append(name)
        buf.clear()

    for fname, raw in iter_jsons(Path(args.src)):
        buf.append((fname, raw))
        n_files += 1
        if len(buf) >= args.per_shard:
            flush()
            print(f"  ... {n_files} files -> {len(shard_names)} shards", flush=True)
    if buf:
        flush()
        print(f"  ... {n_files} files -> {len(shard_names)} shards", flush=True)

    (out / "shards_manifest.json").write_text(json.dumps({
        "shards": shard_names,
        "total_files": n_files,
        "format": "moi dong: {file, data}; data = noi dung JSON goc",
        "edge_contract": ["AST", "CFG", "ARGUMENT", "TARG", "PARAMETER_OUT", "EVAL_ORDER"],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[shards] {n_files} files -> {len(shard_names)} shards tai {out}")


if __name__ == "__main__":
    main()
