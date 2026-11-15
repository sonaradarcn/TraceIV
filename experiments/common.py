"""Helpers shared by the experiment scripts: command line, output location, result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_instruments.config import default_device
from trace_instruments.paths import results_dir


def make_parser(description: str, results_subdir, device: str | None = "auto") -> argparse.ArgumentParser:
    """Parser with the arguments every experiment has.

    ``--output-dir`` defaults to ``<results root>/<results_subdir>`` (see ``trace_instruments.paths``).
    ``--device`` defaults to ``device``; ``"auto"`` is the first GPU when there is one and the CPU
    otherwise, and ``None`` omits the argument (experiments that do not use torch).
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=results_dir(*results_subdir), help="directory of the result files"
    )
    if device is not None:
        default = default_device() if device == "auto" else device
        parser.add_argument("--device", default=default, help="torch device, e.g. cpu or cuda:0")
    return parser


def stored_arguments(args: argparse.Namespace) -> dict:
    """The arguments recorded in a result file: everything except the output location."""
    return {key: value for key, value in vars(args).items() if key != "output_dir"}


def write_json(path: Path, record, **json_options) -> None:
    """Write a result file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(record, handle, **json_options)


def float_list(text: str) -> list[float]:
    """``"0.05,0.1"`` -> ``[0.05, 0.1]``."""
    return [float(item) for item in text.split(",")]


def int_list(text: str) -> list[int]:
    """``"4,10"`` -> ``[4, 10]``."""
    return [int(item) for item in text.split(",")]


def family_tag(family: str) -> str:
    """File-name form of a matrix family (``schur:1.0`` -> ``schur1.0``)."""
    return family.replace(":", "")
