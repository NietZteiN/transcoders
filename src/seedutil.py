"""Deterministic seeding — one place, called at the top of every run.

../CLAUDE.md §4: "Set and record a random seed for every run." `set_seed` seeds Python,
NumPy, and (lazily) torch + CUDA, and requests deterministic kernels. The seed that was
applied is returned so the caller can record it in the run manifest.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed all RNGs in play. Returns the seed (for provenance logging).

    torch is imported lazily so this module stays torch-free at import time.
    """
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
        if deterministic:
            # Reproducibility over raw speed for research runs.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

    return seed
