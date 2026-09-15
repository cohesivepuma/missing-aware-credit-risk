"""Seed Python, NumPy and PyTorch consistently."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed random generators and request deterministic PyTorch operations.

    Identical hardware/software are needed for bitwise reproducibility.
    PYTHONHASHSEED must be set before launching Python to affect hash order;
    setting it here also propagates the value to child processes.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
