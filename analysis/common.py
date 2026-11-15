"""Loading of result files."""

from __future__ import annotations

import json
from pathlib import Path

from trace_instruments.paths import results_dir


def result_files(*parts: str, pattern: str = "*.json") -> list[Path]:
    """Sorted result files below ``results/<parts...>`` that match ``pattern``."""
    return sorted(results_dir(*parts).glob(pattern))


def load(path: Path):
    with open(path) as handle:
        return json.load(handle)


def load_all(*parts: str, pattern: str = "*.json") -> list:
    """Contents of all matching result files, in file-name order."""
    return [load(path) for path in result_files(*parts, pattern=pattern)]


# Order in which the families of recurrent matrices are reported and drawn.
FAMILY_ORDER = ("chain", "schur", "normal", "orthogonal", "random")


def family_sort_key(record):
    """Sort key of a teacher-student record: family in ``FAMILY_ORDER``, then its parameter, then the gain."""
    name, _, parameter = record["family"].partition(":")
    return FAMILY_ORDER.index(name), float(parameter or 0.0), record["gain"]


def load_by_family(*parts: str, pattern: str = "*.json") -> list:
    """Records of all matching result files, ordered by matrix family rather than by file name."""
    return sorted(load_all(*parts, pattern=pattern), key=family_sort_key)


def load_optional(*parts: str):
    """Content of one result file, or ``None`` when it does not exist."""
    path = results_dir(*parts)
    return load(path) if path.exists() else None
