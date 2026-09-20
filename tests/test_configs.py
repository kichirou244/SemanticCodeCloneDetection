"""Test config YAML nạp được và chứa khóa bắt buộc."""

from pathlib import Path

from src.common.config import load_config

REQUIRED = {"dataset", }


def test_dataset_configs():
    for name in ["poj104", "codenet_python"]:
        cfg = load_config(Path("configs/datasets") / f"{name}.yaml")
        assert "dataset" in cfg and "splitting" in cfg
