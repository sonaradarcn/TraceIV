"""Regression tests: the experiment entry points at a tiny size against stored reference outputs.

The reference files in ``tests/reference/<case>/`` were written by the code as it was when the results of
the paper were produced, with the arguments listed in ``tests/reference/cases.json``; the two cases
added in revision (``tracking_lag``, ``online_prediction_power``) have no earlier code and their
references were written by this code.  A run here must
reproduce every number in them; the tolerance only allows for floating-point differences between
platforms and library versions.  With ``TRACE_INSTRUMENTS_EXACT=1`` the tolerance is zero: on the reference
platform (Windows, Python 3.11, the versions of ``requirements.txt``, CPU) every number is bit-identical.
The exact rational-arithmetic experiment is compared with the shipped result file of the paper, with no
tolerance.  Two entry points have no reference run: ``closed_form_checks``, whose fixed sweep takes
minutes (its closed forms are tested in ``test_linear_theory.py``), and ``step_cost_benchmark``, which
measures wall time; both are exercised by the demo.
"""

import importlib
import json
import math
import os
import shlex
from pathlib import Path

import pytest

from trace_instruments.paths import REPOSITORY_ROOT

REFERENCE = Path(__file__).parent / "reference"
CASES = json.loads((REFERENCE / "cases.json").read_text())
NOT_COMPARED = {"args", "seconds"}  # the command line and the wall time
RTOL, ATOL = (0.0, 0.0) if os.environ.get("TRACE_INSTRUMENTS_EXACT") == "1" else (1e-6, 1e-10)


def differences(reference, value, rtol, atol, path=""):
    """Paths at which two nested JSON values differ."""
    if isinstance(reference, dict) and isinstance(value, dict):
        out = []
        for key in sorted(set(reference) | set(value)):
            if key in NOT_COMPARED:
                continue
            if key not in reference or key not in value:
                out.append(f"{path}/{key}: missing on one side")
            else:
                out += differences(reference[key], value[key], rtol, atol, f"{path}/{key}")
        return out
    if isinstance(reference, list) and isinstance(value, list):
        if len(reference) != len(value):
            return [f"{path}: length {len(reference)} != {len(value)}"]
        return [
            d for i, (r, v) in enumerate(zip(reference, value)) for d in differences(r, v, rtol, atol, f"{path}[{i}]")
        ]
    if isinstance(reference, float) and isinstance(value, float):
        if math.isnan(reference) and math.isnan(value):
            return []
        if abs(reference - value) <= atol + rtol * abs(reference):
            return []
        return [f"{path}: {reference!r} != {value!r}"]
    return [] if reference == value else [f"{path}: {reference!r} != {value!r}"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_entry_point_reproduces_reference(case, tmp_path):
    module = importlib.import_module(case["module"])
    module.main(shlex.split(case["arguments"]) + ["--output-dir", str(tmp_path), "--device", "cpu"])
    reference_files = sorted(path.name for path in (REFERENCE / case["id"]).glob("*.json"))
    assert reference_files, "no reference files"
    assert sorted(path.name for path in tmp_path.glob("*.json")) == reference_files
    for name in reference_files:
        found = differences(
            json.loads((REFERENCE / case["id"] / name).read_text()),
            json.loads((tmp_path / name).read_text()),
            RTOL,
            ATOL,
        )
        assert not found, f"{name}: {len(found)} differences, first: {found[:5]}"


def test_exact_example_matches_shipped_result(tmp_path):
    """Rational arithmetic: the determinant and the characteristic polynomial are reproduced exactly."""
    module = importlib.import_module("experiments.linear_dynamics.snap1_exact_instability")
    module.main(["--output-dir", str(tmp_path)])
    name = "snap1_exact_instability.json"
    shipped = json.loads((REPOSITORY_ROOT / "results" / "linear_dynamics" / name).read_text())
    new = json.loads((tmp_path / name).read_text())
    for key in ("matrix", "characteristic_polynomial", "det_snap1", "det_immediate"):
        assert new[key] == shipped[key]
    assert not differences(shipped, new, 1e-9, 1e-9)
    assert new["det_snap1_float"] < 0 < new["det_immediate_float"]
