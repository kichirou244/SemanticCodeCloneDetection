"""Seed tái lập dùng chung cho toàn project (numpy / random / torch)."""

import os
import random


def set_global_seed(seed: int) -> None:
    """Cố định seed cho `random`, `numpy` và `torch` (nếu có)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
