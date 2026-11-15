"""Numerical conventions shared by the whole code base."""

from __future__ import annotations

import torch

#: All computation is carried out in double precision.
DTYPE = torch.float64

#: A lane whose largest parameter exceeds this bound (or is not finite) is counted as diverged.
DIVERGENCE_BOUND = 1e3


def default_device() -> str:
    """First CUDA device when one is available, the CPU otherwise."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"
