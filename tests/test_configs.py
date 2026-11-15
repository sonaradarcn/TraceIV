"""The run lists in ``configs/`` against the entry points and the shipped result files.

Every listed invocation must be accepted by the parser of its entry point; every result file of the
paper must exist, be recorded in ``results/runs_manifest.json`` under the invocation that the paper
configuration lists, and store exactly the arguments that this invocation resolves to.
"""

import argparse
import importlib
import json
import shlex

import pytest

from experiments.registry import BY_NAME, EXPERIMENTS
from experiments.reproduce import NO_DEVICE_ARGUMENT, load_runs, signature
from trace_instruments.paths import CONFIG_DIR, REPOSITORY_ROOT

RESULTS = REPOSITORY_ROOT / "results"
PAPER_RUNS = load_runs(CONFIG_DIR / "paper_runs.json")
DEMO_RUNS = load_runs(CONFIG_DIR / "demo_runs.json")
NOT_STORED = {"output_dir", "device"}  # the location, and the device (recorded, but machine-specific)


class _Parsed(Exception):
    pass


def resolved_arguments(run, monkeypatch):
    """Namespace the entry point of ``run`` obtains from its arguments, without running the experiment."""
    original = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise _Parsed(vars(original(self, args, namespace)))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    module = importlib.import_module(BY_NAME[run["experiment"]].module)
    arguments = shlex.split(run["arguments"])
    if run["experiment"] not in NO_DEVICE_ARGUMENT:
        arguments += ["--device", "cpu"]
    with pytest.raises(_Parsed) as caught:
        module.main(arguments)
    return caught.value.args[0]


def run_id(run):
    return f"{run['experiment']}:{run['outputs'][0]}"


def test_every_experiment_is_listed():
    assert {run["experiment"] for run in PAPER_RUNS} == {experiment.name for experiment in EXPERIMENTS}
    assert {run["experiment"] for run in DEMO_RUNS} == {experiment.name for experiment in EXPERIMENTS}


@pytest.mark.parametrize("run", DEMO_RUNS, ids=run_id)
def test_demo_invocation_is_accepted(run, monkeypatch):
    resolved_arguments(run, monkeypatch)


@pytest.mark.parametrize("run", PAPER_RUNS, ids=run_id)
def test_shipped_results_belong_to_the_paper_invocation(run, monkeypatch):
    expected = {key: value for key, value in resolved_arguments(run, monkeypatch).items() if key not in NOT_STORED}
    manifest = json.loads((RESULTS / "runs_manifest.json").read_text())
    for output in run["outputs"]:
        assert (RESULTS / output).exists(), f"{output} is missing"
        assert manifest.get(output) == signature(run), f"{output}: manifest names another invocation"
        stored = json.loads((RESULTS / output).read_text()).get("args")
        if stored is None:  # the exact rational example takes no arguments and stores none
            assert run["experiment"] == "snap1_exact_instability"
            continue
        stored = {key: value for key, value in stored.items() if key not in NOT_STORED}
        assert stored == json.loads(json.dumps(expected, default=str)), f"{output}: stored arguments differ"


def test_no_result_file_is_unlisted():
    listed = {output for run in PAPER_RUNS for output in run["outputs"]}
    shipped = {path.relative_to(RESULTS).as_posix() for path in RESULTS.rglob("*.json")} - {
        "runs_manifest.json",
        "runs_checksums.json",
    }
    assert shipped == listed


def test_cpu_fallback_is_a_different_invocation():
    """A run made on a GPU that falls back to the CPU must not be taken for the GPU run."""
    run = next(run for run in load_runs(CONFIG_DIR / "paper_runs.json") if run["device"] == "cuda")
    assert signature(run) == signature(run, "cuda")
    assert signature(run, "cpu") != signature(run, "cuda")


def test_real_series_ignore_the_length_arguments():
    """The real records are split 80/20 whatever lengths are asked for; seeds of ``power`` share no data."""
    import numpy as np

    from trace_instruments.prediction_streams import POWER, make_series

    short, long = make_series(POWER, 0, 10, 5), make_series(POWER, 0, 50_000, 10_000)
    assert [len(part) for part in short] == [6399, 6399, 1600, 1600]
    assert all(np.array_equal(a, b) for a, b in zip(short, long))
    assert [len(part) for part in make_series("sunspot", 0, 10, 5)] == [2662, 2662, 666, 666]
    assert not np.array_equal(make_series(POWER, 1, 0, 0)[0], short[0])


def test_shipped_results_have_their_recorded_checksums():
    """The runner skips a run only when its result files are the ones it recorded."""
    from experiments.reproduce import CHECKSUMS, checksum

    recorded = json.loads((RESULTS / CHECKSUMS).read_text())
    manifest = json.loads((RESULTS / "runs_manifest.json").read_text())
    assert set(recorded) == set(manifest)
    for output, digest in recorded.items():
        assert checksum(RESULTS / output) == digest, f"{output} differs from the file recorded by the runner"


def test_real_series_accept_any_eval_length(tmp_path, monkeypatch, capsys):
    """The burn-in check of ``--eval-length`` concerns the synthetic tasks only."""
    from experiments.prediction_tasks import online_prediction

    with pytest.raises(SystemExit):
        online_prediction.main(["--tasks", "delay:5,sunspot", "--eval-length", "100"])
    assert "--eval-length must exceed" in capsys.readouterr().err
    monkeypatch.setenv("TRACE_INSTRUMENTS_RESULTS_DIR", str(tmp_path))
    arguments = "--tasks sunspot --eval-length 100 --num-units 4 --num-seeds 1 --step-sizes 0.001 --device cpu"
    online_prediction.main(arguments.split() + ["--training-steps", "300", "--handover-steps", "100"])
    assert len(list(tmp_path.rglob("*sunspot.json"))) == 1


def fake_run(root):
    """Stand-in for the subprocess of the runner: writes the result files named after ``--marker``."""

    def run(cmd, cwd=None):
        for name in ("a.json", "b.json"):
            (root / name).write_text(json.dumps({"written_by": cmd[-1]}))
        return argparse.Namespace(returncode=0)

    return run


def test_runner_refuses_an_altered_result_file(tmp_path, monkeypatch, capsys):
    """First run writes and records; second is skipped; a changed file is refused; ``--force`` reruns."""
    from experiments import reproduce

    monkeypatch.setenv(reproduce.RESULTS_ENV, str(tmp_path))  # restored after the test
    config = tmp_path / "runs.json"
    run = {"experiment": "tracking_lag", "device": "cpu", "arguments": "--seed 0", "outputs": ["a.json", "b.json"]}
    config.write_text(json.dumps({"runs": [run]}))
    root = tmp_path / "results"
    root.mkdir()
    calls = []
    monkeypatch.setattr(
        reproduce.subprocess, "run", lambda cmd, cwd=None: calls.append(cmd) or fake_run(root)(cmd, cwd)
    )
    arguments = ["--config", str(config), "--results-dir", str(root)]

    reproduce.main(arguments)
    assert len(calls) == 1 and set(json.loads((root / reproduce.CHECKSUMS).read_text())) == {"a.json", "b.json"}
    reproduce.main(arguments)
    assert len(calls) == 1 and "skipped" in capsys.readouterr().out

    (root / "b.json").write_text(json.dumps({"written_by": "hand"}))
    with pytest.raises(SystemExit) as stop:
        reproduce.main(arguments)
    assert len(calls) == 1 and stop.value.code and "NOT RUN" in capsys.readouterr().out
    assert json.loads((root / "b.json").read_text()) == {"written_by": "hand"}

    reproduce.main(arguments + ["--force"])
    assert len(calls) == 2 and json.loads((root / "b.json").read_text()) != {"written_by": "hand"}
    reproduce.main(arguments)
    assert len(calls) == 2


def test_list_sets_the_results_directory_before_the_commands(tmp_path, monkeypatch, capsys):
    """Commands copied from ``--list`` must not write to results/ when another directory was asked for."""
    from experiments import reproduce

    monkeypatch.setenv(reproduce.RESULTS_ENV, str(tmp_path))  # restored after the test
    reproduce.main(["--only", "tracking_lag", "--results-dir", str(tmp_path / "my results"), "--list"])
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    export = next(i for i, line in enumerate(lines) if line.startswith(f"export {reproduce.RESULTS_ENV}="))
    first_command = next(i for i, line in enumerate(lines) if line.startswith("python -m"))
    assert export < first_command and shlex.split(lines[export])[1].endswith("my results")
