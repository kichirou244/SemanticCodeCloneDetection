"""Script trích xuất và đóng gói đặc trưng AST (M3 ASTNN) và CPG (M4 CPG-GNN) cho BigCloneBench Java.

Hỗ trợ chạy trên cả môi trường local (với tập smoke/dev) và Kaggle GPU/CPU:
  1. Tải hoặc nạp các cặp mã nguồn BigCloneBench.
  2. Gom cụm và trích xuất danh sách các hàm Java duy nhất (unique functions) dựa trên func_id (id1, id2).
     -> Giảm từ ~1.73 triệu lượt gọi hàm xuống chỉ còn ~25k - 50k hàm duy nhất.
  3. AST Cache (M3): Parse cây cú pháp bằng tree-sitter-java, compile câu lệnh và lưu vào bcb_ast_cache.npz.
  4. CPG Export (M4): Tạo wrapper class cho các hàm đơn lẻ, chạy Joern javasrc2cpg theo lô và xuất JSON.
  5. Đóng gói Kaggle Dataset: Nén cpg_export.tar.gz, sinh dataset-metadata.json để tạo dataset trên Kaggle chỉ bằng 1 click.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import tarfile
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.bigclonebench import BigCloneBenchLoader
from src.representations.ast.store import ASTTreeStore
from scripts.build_cpg_export import (
    joern_or_die,
    run_batch,
    run_smoke,
    SMOKE_CODE_JAVA,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_bcb_features")


def collect_unique_functions(
    loader: BigCloneBenchLoader,
    data_file: Path | None = None,
    split: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Gom các hàm Java duy nhất từ các cặp BigCloneBench."""
    t0 = time.perf_counter()
    pairs = []
    if data_file is not None and Path(data_file).exists():
        p_path = Path(data_file)
        logger.info("Đang nạp dữ liệu từ file local: %s ...", p_path)
        if p_path.suffix == ".parquet":
            pairs = loader.load_parquet(p_path, max_samples=limit)
        else:
            pairs = loader.load_jsonl(p_path, max_samples=limit)
    else:
        # Nạp từ Hugging Face
        splits = [split] if split and split != "all" else ["train", "validation", "test"]
        for s in splits:
            hf_s = "validation" if s == "val" else s
            logger.info("Đang nạp dữ liệu split '%s' từ Hugging Face (limit=%s)...", hf_s, limit)
            split_pairs = loader.load_from_hf(split=hf_s, max_samples=limit)
            pairs.extend(split_pairs)

    unique_funcs: dict[str, str] = {}
    for p in pairs:
        sid1 = str(p.id1)
        sid2 = str(p.id2)
        if sid1 not in unique_funcs and p.func1:
            unique_funcs[sid1] = p.func1
        if sid2 not in unique_funcs and p.func2:
            unique_funcs[sid2] = p.func2

    el = time.perf_counter() - t0
    stats = {
        "total_pairs": len(pairs),
        "unique_functions": len(unique_funcs),
        "reduction_ratio": round(len(pairs) * 2 / max(len(unique_funcs), 1), 2),
        "elapsed_seconds": round(el, 2),
    }
    logger.info(
        "Đã thu thập %d hàm duy nhất từ %d cặp (tỉ lệ nén: %.1fx, mất %.1fs)",
        stats["unique_functions"],
        stats["total_pairs"],
        stats["reduction_ratio"],
        stats["elapsed_seconds"],
    )
    return unique_funcs, stats


def build_ast_cache(
    unique_funcs: dict[str, str],
    out_dir: Path,
    language: str = "java",
) -> Path:
    """Xây dựng và lưu kho cache AST cây câu lệnh cho M3 ASTNN."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "bcb_ast_cache.npz"
    logger.info("--- BẮT ĐẦU XÂY DỰNG AST CACHE CHO M3 (ASTNN) ---")
    t0 = time.perf_counter()
    store = ASTTreeStore.build(unique_funcs, language=language, cache_path=cache_path, verbose=True)
    el = time.perf_counter() - t0
    logger.info(
        "Hoàn tất AST cache: %d hàm -> %s (%.1f MiB, mất %.1fs, tốc độ %.0f hàm/s)",
        len(store),
        cache_path.name,
        cache_path.stat().st_size / (1024 * 1024),
        el,
        len(store) / max(el, 1e-6),
    )
    return cache_path


def build_cpg_cache(
    unique_funcs: dict[str, str],
    out_dir: Path,
    work_dir: Path,
    joern_cmd: str,
    batch_size: int = 100,
    timeout_s: int = 300,
    archive: bool = True,
) -> tuple[Path, dict[str, Any]]:
    """Xuất đồ thị CPG bằng Joern javasrc2cpg cho M4 CPG-GNN."""
    logger.info("--- BẮT ĐẦU TRÍCH XUẤT CPG BẰNG JOERN CHO M4 (CPG-GNN) ---")
    cpg_out = out_dir / "cpg_export"
    cpg_out.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Chuyển đổi sang list samples tương thích với build_cpg_export
    samples = [{"sample_id": fid, "code": code} for fid, code in unique_funcs.items()]
    todo = [s for s in samples if not (cpg_out / f"{s['sample_id']}.json").exists()]
    logger.info("Tổng số hàm: %d | Còn lại cần xuất: %d", len(samples), len(todo))

    t0 = time.perf_counter()
    n_ok = 0
    n_err = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        ok, err = run_batch(
            batch,
            cpg_out,
            work_dir,
            joern_cmd,
            work_dir / f"batch_{i:06d}",
            timeout_s,
            language="java",
        )
        n_ok += ok
        n_err += err
        el = time.perf_counter() - t0
        done = i + len(batch)
        rate = n_ok / max(el, 1e-6)
        logger.info(
            "[cpg-export] Đã xử lý %d/%d (ok=%d, err=%d, %.1f phút, %.2f hàm/s)",
            done,
            len(todo),
            n_ok,
            n_err,
            el / 60,
            rate,
        )

    # Nén tar.gz nếu được yêu cầu
    tar_path = out_dir / "bcb_cpg_export.tar.gz"
    if archive and list(cpg_out.glob("*.json")):
        logger.info("Đang nén các file CPG JSON thành %s để tối ưu Kaggle Dataset...", tar_path.name)
        with tarfile.open(tar_path, "w:gz") as tar:
            for json_file in cpg_out.glob("*.json"):
                tar.add(json_file, arcname=f"cpg_export/{json_file.name}")
        logger.info("Đã tạo archive: %s (%.1f MiB)", tar_path.name, tar_path.stat().st_size / (1024 * 1024))

    stats = {
        "total_requested": len(samples),
        "total_ok": n_ok + (len(samples) - len(todo)),
        "total_err": n_err,
        "export_dir": str(cpg_out),
        "archive_file": str(tar_path) if archive else None,
    }
    return cpg_out, stats


def create_kaggle_dataset_metadata(out_dir: Path, dataset_slug: str = "bigclonebench-features") -> Path:
    """Tạo metadata dataset cho Kaggle để có thể upload trực tiếp bằng Kaggle API hoặc giao diện web."""
    meta = {
        "title": "BigCloneBench Extracted Features (AST & CPG)",
        "id": f"{dataset_slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": "Pre-extracted AST statement trees (for M3 ASTNN) and Code Property Graphs (for M4 CPG-GNN) on BigCloneBench Java code pairs.",
    }
    meta_path = out_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="BigCloneBench AST & CPG Feature Pre-extraction")
    parser.add_argument("--data-file", type=str, default=None, help="Đường dẫn file parquet/jsonl local")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "validation", "test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số cặp đọc (dùng khi thử nghiệm)")
    parser.add_argument("--features", type=str, default="both", choices=["ast", "cpg", "both"])
    parser.add_argument("--out-dir", type=str, default="artifacts/bcb_features", help="Thư mục xuất kết quả")
    parser.add_argument("--work-dir", type=str, default="artifacts/cpg_work", help="Thư mục làm việc tạm")
    parser.add_argument("--batch-size", type=int, default=100, help="Số hàm per Joern batch")
    parser.add_argument("--joern", type=str, default=None, help="Lệnh hoặc đường dẫn joern binary")
    parser.add_argument("--timeout-s", type=int, default=300, help="Timeout mỗi lô Joern (giây)")
    parser.add_argument("--archive", action="store_true", default=True, help="Nén CPG JSON thành file .tar.gz")
    parser.add_argument("--dataset-slug", type=str, default="bigclonebench-features", help="Tên slug dataset Kaggle")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)

    loader = BigCloneBenchLoader()
    unique_funcs, p_stats = collect_unique_functions(loader, data_file=args.data_file, split=args.split, limit=args.limit)

    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": "java",
        "pair_stats": p_stats,
    }

    # 1. Trích xuất AST
    if args.features in ("ast", "both"):
        ast_path = build_ast_cache(unique_funcs, out_dir, language="java")
        manifest["ast_cache"] = {
            "path": str(ast_path),
            "size_bytes": ast_path.stat().st_size,
        }

    # 2. Trích xuất CPG bằng Joern
    if args.features in ("cpg", "both"):
        joern_cmd = args.joern or joern_or_die()
        logger.info("Kiểm tra Joern smoke trên Java...")
        run_smoke(SMOKE_CODE_JAVA, work_dir, joern_cmd, timeout_s=args.timeout_s, language="java")
        cpg_dir, cpg_stats = build_cpg_cache(
            unique_funcs,
            out_dir,
            work_dir,
            joern_cmd,
            batch_size=args.batch_size,
            timeout_s=args.timeout_s,
            archive=args.archive,
        )
        manifest["cpg_export"] = cpg_stats

    # 3. Tạo Kaggle Dataset Metadata
    meta_path = create_kaggle_dataset_metadata(out_dir, dataset_slug=args.dataset_slug)
    manifest["kaggle_metadata"] = str(meta_path)

    # Lưu manifest
    manifest_path = out_dir / "bcb_features_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Đã hoàn tất trích xuất đặc trưng! Manifest lưu tại: %s", manifest_path)


if __name__ == "__main__":
    main()
