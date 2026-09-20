"""Build dataset Phase 1: clean → normalize → dedup → split → pairs → leakage.

Chạy:
  python scripts/build_dataset.py --config configs/pairs/poj104.yaml
  python scripts/build_dataset.py --config configs/pairs/codenet_python.yaml

Scripts chỉ orchestrate components trong src/data (không logic trực tiếp).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.seed import set_global_seed
from src.data.cleaning.cleaner import CleaningConfig, DatasetCleaner
from src.data.deduplication.groups import DuplicateDetector
from src.data.leakage.checks import run_pair_leakage_audit
from src.data.loaders.codenet_python import CodeNetPythonLoader
from src.data.loaders.poj104 import POJ104Loader
from src.data.manifest import build_manifest, save_manifest
from src.data.normalization.normalizer import CodeNormalizer
from src.data.pairs.generator import PairGenConfig, PairGenerator
from src.data.splitting.problem_split import (
    filter_eligible,
    save_problem_ids,
    split_by_problem,
)

LOADERS = {"poj104": POJ104Loader, "codenet_python": CodeNetPythonLoader}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds_name = cfg["dataset"]["name"]
    seed = cfg["splitting"]["seed"]
    set_global_seed(seed)

    # 1) Load
    loader_kw = {}
    if ds_name == "codenet_python":
        loader_kw = {
            "split": cfg["dataset"].get("split"),
            "streaming": cfg["dataset"].get("streaming", False),
            "max_samples": cfg["dataset"].get("max_samples", 0),
            "seed": seed,
        }
    samples = LOADERS[ds_name]().load(**loader_kw)
    n_raw = len(samples)

    # 2) Clean (mọi record loại đều có reason)
    cc = cfg.get("cleaning", {})
    cleaner = DatasetCleaner(
        CleaningConfig(
            min_code_chars=cc.get("min_code_chars", 10),
            expected_language=cc.get("expected_language", ""),
            status_allowlist=cc.get("status_allowlist"),
        )
    )
    cleaned = cleaner.clean(samples)

    # 3) Normalize (giữ raw, chỉ điền normalized_code cho dedup/leakage)
    normalizer = CodeNormalizer()
    for s in cleaned.kept:
        s.normalized_code = normalizer.normalize(s.code)

    # 4) Dedup TRƯỚC split (DEC-010): exact + normalized, canonical deterministic
    dedup = DuplicateDetector().detect(cleaned.kept)

    # 5) Lọc problems đủ điều kiện + problem split
    kept, ineligible = filter_eligible(
        dedup.canonical, cfg["splitting"].get("min_samples_per_problem", 2)
    )
    # Danh sách ineligible có thể hàng trăm mục → lưu file riêng, manifest chỉ giữ số lượng.
    ineligible_path = Path(cfg["output"]["splits_dir"]) / "ineligible_problems.json"
    ineligible_path.parent.mkdir(parents=True, exist_ok=True)
    ineligible_path.write_text(json.dumps(ineligible, indent=2), encoding="utf-8")
    splits = split_by_problem(
        kept,
        train_ratio=cfg["splitting"].get("train_ratio", 0.8),
        val_ratio=cfg["splitting"].get("val_ratio", 0.1),
        seed=seed,
    )
    split_paths = save_problem_ids(splits, cfg["output"]["splits_dir"])

    # 6) Pair generation từng split
    pg = cfg.get("pair_generation", {})
    gen = PairGenerator(
        PairGenConfig(
            seed=pg.get("seed", seed),
            negative_ratio=pg.get("negative_ratio", 1),
            max_positive_pairs_per_problem=pg.get("max_positive_pairs_per_problem", 2000),
            max_positive_pairs_per_problem_val_test=pg.get(
                "max_positive_pairs_per_problem_val_test", 500
            ),
        )
    )
    pair_splits, pair_paths = {}, {}
    for split, subset in splits.items():
        pairs = gen.generate(subset, split)
        pair_splits[split] = pairs
        path = Path(cfg["output"]["pairs_dir"]) / f"{split}.jsonl"
        write_jsonl(path, [p.__dict__ for p in pairs])
        pair_paths[split] = str(path)

    # 7) Leakage audit cuối (sample + code + normalized + problem + pair)
    audit = run_pair_leakage_audit(splits, pair_splits)
    print(json.dumps(
        {k: (v if k != "pair" else v) for k, v in audit.items()}, indent=2
    ))
    if not audit["passed"]:
        print("LEAKAGE FAILED — dataset không được sử dụng.")
        raise SystemExit(1)

    # 8) Lưu processed samples + manifest
    proc_path = Path(cfg["output"]["processed"])
    write_jsonl(
        proc_path,
        [
            {
                "sample_id": s.sample_id,
                "code": s.code,
                "problem_id": s.problem_id,
                "split": s.split,
            }
            for split_samples in splits.values()
            for s in split_samples
        ],
    )
    pair_stats = {
        split: {
            "positive": sum(1 for p in pairs if p.label == 1),
            "negative": sum(1 for p in pairs if p.label == 0),
        }
        for split, pairs in pair_splits.items()
    }
    manifest = build_manifest(
        dataset=ds_name,
        language=cfg["dataset"]["language"],
        seed=seed,
        split_strategy=cfg["splitting"].get("strategy", "problem"),
        cleaning=cleaned.summary(),
        dedup=dedup.summary(),
        splits={
            split: {
                "n_samples": len(v),
                "n_problems": len({s.problem_id for s in v}),
            }
            for split, v in splits.items()
        },
        pairs=pair_stats,
        artifacts={
            "n_raw": n_raw,
            "ineligible_problems": {"n": len(ineligible), "file": str(ineligible_path)},
            "split_files": split_paths,
            "pair_files": pair_paths,
            "processed": str(proc_path),
        },
        config_snapshot=args.config,
    )
    manifest["leakage"] = {k: v.get("ok", v) for k, v in audit.items() if k != "passed"}
    manifest["leakage_passed"] = audit["passed"]
    save_manifest(manifest, cfg["output"]["manifest"])
    print(f"manifest: {cfg['output']['manifest']}")


if __name__ == "__main__":
    main()
