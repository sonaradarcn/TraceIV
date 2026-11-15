"""Locations of data, results and figures.

Every script resolves its paths through this module, so the layout is defined in one place.  Two
environment variables relocate the outputs without touching the code:

``TRACE_INSTRUMENTS_RESULTS_DIR``  root of the JSON result files   (default ``<repository>/results``)
``TRACE_INSTRUMENTS_FIGURES_DIR``  where the figures are written    (default ``<repository>/figures``)
"""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPOSITORY_ROOT / "data"
CONFIG_DIR = REPOSITORY_ROOT / "configs"


def results_root() -> Path:
    """Root directory of the result files."""
    return Path(os.environ.get("TRACE_INSTRUMENTS_RESULTS_DIR", REPOSITORY_ROOT / "results"))


def figures_dir() -> Path:
    """Directory the figures are written to."""
    return Path(os.environ.get("TRACE_INSTRUMENTS_FIGURES_DIR", REPOSITORY_ROOT / "figures"))


def results_dir(*parts: str) -> Path:
    """Sub-directory of the results root, e.g. ``results_dir("restoration", "exchange_rate")``."""
    return results_root().joinpath(*parts)
