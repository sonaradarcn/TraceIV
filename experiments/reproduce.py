"""Run the experiments of the paper from the list of invocations in ``configs/paper_runs.json``.

    python -m experiments.reproduce --list                       # what would run, and where each result goes
    python -m experiments.reproduce                              # everything (about 60 hours on the reference machine)
    python -m experiments.reproduce --only readout_rank,displacement
    python -m experiments.reproduce --config configs/demo_runs.json --results-dir demo_results     # about ten minutes

Every run is one ``python -m <module> <arguments> --device <device>`` in a subprocess, so a line printed by
``--list`` can also be pasted into a shell.  Runs marked ``cuda`` in the configuration were made on a GPU;
they use ``--cuda-device`` and fall back to the CPU (with a warning) when there is none.

``<results dir>/runs_manifest.json`` records the invocation that produced each result file and
``runs_checksums.json`` the SHA-256 of the file as written.  A run is skipped only if all its result files
exist, were produced by the same invocation and still have the recorded checksum; files left by a
different invocation, or changed since, are never overwritten silently: the run is reported and needs
``--force`` or another ``--results-dir``.  The manifest does not record the version of the code: after
changing the code or its dependencies, rerun with ``--force`` or into a new ``--results-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from experiments.registry import BY_NAME
from trace_instruments.paths import CONFIG_DIR, REPOSITORY_ROOT, results_root

RESULTS_ENV = "TRACE_INSTRUMENTS_RESULTS_DIR"
NO_DEVICE_ARGUMENT = {"closed_form_checks", "snap1_exact_instability"}  # numpy / sympy only
MANIFEST = "runs_manifest.json"
CHECKSUMS = "runs_checksums.json"


def load_runs(path: Path) -> list[dict]:
    with open(path) as handle:
        runs = json.load(handle)["runs"]
    for run in runs:
        if run["experiment"] not in BY_NAME:
            raise SystemExit(f"{path}: unknown experiment {run['experiment']!r}")
    return runs


def cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def command(run: dict, cuda_device: str) -> list[str]:
    arguments = shlex.split(run["arguments"])
    if run["experiment"] not in NO_DEVICE_ARGUMENT:
        arguments += ["--device", cuda_device if run["device"] == "cuda" else "cpu"]
    return [sys.executable, "-m", BY_NAME[run["experiment"]].module, *arguments]


def signature(run: dict, device_kind: str | None = None) -> str:
    """What identifies an invocation: module, arguments and the kind of device it runs on.

    ``device_kind`` is the kind actually used (``cpu`` when a run made on a GPU falls back to the CPU);
    by default it is the kind the configuration asks for."""
    kind = device_kind or run["device"]
    return f"{BY_NAME[run['experiment']].module} {' '.join(shlex.split(run['arguments']))} [{kind}]"


def checksum(path: Path) -> str:
    """SHA-256 of a result file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(root: Path) -> dict:
    path = root / CHECKSUMS
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def write_checksums(root: Path, checksums: dict) -> None:
    with open(root / CHECKSUMS, "w") as handle:
        json.dump(dict(sorted(checksums.items())), handle, indent=1)


def read_manifest(root: Path) -> dict:
    path = root / MANIFEST
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def write_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with open(root / MANIFEST, "w") as handle:
        json.dump(dict(sorted(manifest.items())), handle, indent=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "paper_runs.json", help="list of runs")
    parser.add_argument("--only", default="", help="comma-separated experiment names (see experiments.registry)")
    parser.add_argument("--results-dir", type=Path, help="root of the result files (default: results/)")
    parser.add_argument("--cuda-device", default="cuda:0", help="device of the runs made on a GPU")
    parser.add_argument("--force", action="store_true", help="run even if the result files exist")
    parser.add_argument("--list", action="store_true", help="print the commands and stop")
    args = parser.parse_args(argv)

    if args.results_dir:
        os.environ[RESULTS_ENV] = str(args.results_dir.resolve())
    root = results_root()
    runs = load_runs(args.config)
    if args.only:
        wanted = args.only.split(",")
        unknown = [name for name in wanted if name not in BY_NAME]
        if unknown:
            parser.error(f"unknown experiment(s) {', '.join(unknown)}; choose from {', '.join(BY_NAME)}")
        runs = [run for run in runs if run["experiment"] in wanted]

    cuda_device = args.cuda_device
    if any(run["device"] == "cuda" for run in runs) and not cuda_available():
        print(
            "warning: no GPU found; the runs made on a GPU will use the CPU (slower, same up to rounding) "
            "and are recorded as CPU runs in the manifest"
        )
        cuda_device = "cpu"
    if args.list and args.results_dir:  # a listed command alone would write to results/
        print(f"export {RESULTS_ENV}={shlex.quote(str(args.results_dir.resolve()))}")

    failed = []
    manifest = read_manifest(root)
    checksums = read_checksums(root)
    for index, run in enumerate(runs, start=1):
        cmd = command(run, cuda_device)
        used = "cuda" if run["device"] == "cuda" and cuda_device.startswith("cuda") else "cpu"
        mine = signature(run, used)  # a CPU fallback of a GPU run is a different invocation
        shown = "python " + " ".join(shlex.quote(part) for part in cmd[1:])
        header = f"[{index}/{len(runs)}] {run['experiment']} ({BY_NAME[run['experiment']].paper_items})"
        if args.list:
            minutes = f", about {run['minutes']} min" if "minutes" in run else ""
            print(f"{header}{minutes}\n    {shown}")
            for output in run["outputs"]:
                print(f"    -> {(args.results_dir or Path('results')) / output}")
            continue
        existing = [output for output in run["outputs"] if (root / output).exists()]
        altered = [output for output in existing if checksums.get(output) != checksum(root / output)]
        if not args.force and altered and all(manifest.get(output) == mine for output in existing):
            failed.append(run["experiment"])
            print(
                f"{header}: NOT RUN. {altered[0]} in {root} is not the file this runner recorded "
                "(changed, or written by hand); use another --results-dir, or --force to overwrite",
                flush=True,
            )
            continue
        if not args.force and len(existing) == len(run["outputs"]):
            if all(manifest.get(output) == mine for output in existing):
                print(f"{header}: result files of this invocation exist, skipped (use --force to rerun)", flush=True)
                continue
        if not args.force and any(manifest.get(output) != mine for output in existing):
            failed.append(run["experiment"])
            print(
                f"{header}: NOT RUN. {existing[0]} in {root} comes from a different invocation; "
                "use another --results-dir, or --force to overwrite",
                flush=True,
            )
            continue
        print(f"{header}\n    {shown}", flush=True)
        start = time.time()
        code = subprocess.run(cmd, cwd=REPOSITORY_ROOT).returncode
        missing = [output for output in run["outputs"] if not (root / output).exists()]
        if code or missing:
            failed.append(run["experiment"])
            print(f"    FAILED (exit code {code}; missing {missing})", flush=True)
        else:
            manifest.update({output: mine for output in run["outputs"]})
            write_manifest(root, manifest)
            checksums.update({output: checksum(root / output) for output in run["outputs"]})
            write_checksums(root, checksums)
            print(f"    done in {time.time() - start:.0f} s", flush=True)
    if failed:
        sys.exit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
