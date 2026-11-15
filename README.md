# TraceIV: credit-assignment rules as instruments

Code, result files and figure scripts for the paper

> **Credit-assignment rules as instruments: what an approximate online learning rule converges to, and when**

The paper treats the trace that an approximate online rule propagates in place of the exact sensitivity
(RTRL, SnAp-1, the immediate rule, leaky traces, UORO, truncated BPTT) as an *instrument* for the gradient.
The limit, stability and efficiency of the rule are then read off one matrix, the cross-Gram between the
instrument and the exact sensitivity, and the bias the rule leaves can be estimated online and added back
(*restoration*). This repository contains

- `trace_instruments/` — the library: cells with analytic Jacobians, the trace recursions, online learning
  with restoration, stationary moments and the closed forms for linear dynamics;
- `experiments/` — one entry point per experiment of the paper, grouped by scientific question;
- `analysis/` and `plotting/` — every number and every data figure of the paper, recomputed from result files;
- `results/` — the result files behind the paper (78 JSON files, 11 MB), so that tables and figures can be
  regenerated without rerunning anything.

## Contents

1. [System requirements](#system-requirements)
2. [Installation](#installation)
3. [Demo](#demo)
4. [Reproducing the paper](#reproducing-the-paper)
5. [Map from the paper to the code](#map-from-the-paper-to-the-code)
6. [Repository structure](#repository-structure)
7. [Using the library](#using-the-library)
8. [Result files](#result-files)
9. [Tests](#tests)
10. [Data, citation and licence](#data-citation-and-licence)

## System requirements

| | Tested | Should also work |
|---|---|---|
| Operating system | Windows Server 2022 | Linux, macOS |
| Python | 3.11 | 3.10 or later |
| Packages | torch 2.14.1, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, sympy 1.14.0 | versions in `pyproject.toml` |
| Hardware | 40 CPU cores, one NVIDIA RTX 3080 (20 GB) | any CPU; a GPU is optional |

All experiments run on a CPU. A GPU shortens the largest ones (the prediction tasks and the gated
teacher–student runs); nothing needs more than 2 GB of GPU memory. Everything is computed in double precision.

## Installation

```bash
git clone https://github.com/sonaradarcn/TraceIV.git && cd TraceIV
conda env create -f environment.yml && conda activate trace-instruments   # or: python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # the exact versions used for the paper
pip install -r requirements-dev.txt # optional: pytest and black
```

For a GPU, install the build of `torch` that matches your CUDA version first
(see [pytorch.org](https://pytorch.org/get-started/locally/)). Installation takes a few minutes, almost all
of it the download of `torch`.

The repository is used as a source checkout: every command below is run from its root, where the four
packages are importable as they are and `data/`, `configs/` and `results/` are found next to them.
`pip install -e .` works and is optional. A built wheel does not carry those three directories and is not a
supported way to run the experiments.

## Demo

**Tables and figures from the shipped result files** (under one minute, CPU):

```bash
python -m analysis.paper_numbers          # every number quoted in the paper, section by section
python -m plotting.make_figures           # Figs. 4-9 as PDF in figures/, PNG previews in figures/preview/
```

Expected output: `paper_numbers` prints about 520 lines, among them the rows of Tables 1–3 (see
[the excerpt below](#headline-numbers)); `make_figures` prints `wrote nonnormal`, …, `wrote tasks` and writes
six PDF files, the figures of the paper. The figures are set in Arial; where it is not installed matplotlib
substitutes its default font, which changes the lettering and nothing else.

**Every experiment at a reduced size** (about 10 minutes on a CPU):

```bash
python -m experiments.reproduce --config configs/demo_runs.json --results-dir demo_results
```

This runs all thirteen entry points on small systems and short runs and writes their result files to
`demo_results/`. Each run prints its own summary, for instance for the read-out experiment

```
m=2 (units) P=24 start excess (median) ...
   rtrl                  rank=24/24 ... excess median=...
   snap1                 rank=12/24 rho2min=0.000 ...
   snap1+restoration     rank=12/24 ...
```

(the rank of the SnAp-1 instrument falls to the number of weights that feed read-out units). The numbers of
the demo are noisy and are not those of the paper.

## Reproducing the paper

The exact invocation of every run of the paper is listed in [`configs/paper_runs.json`](configs/paper_runs.json),
and `experiments.reproduce` replays them:

```bash
python -m experiments.reproduce --list                        # the commands, their outputs, their duration
python -m experiments.reproduce --results-dir rerun_results   # everything, into a fresh directory
python -m experiments.reproduce --results-dir rerun_results --only readout_rank    # or one experiment

export TRACE_INSTRUMENTS_RESULTS_DIR="$PWD/rerun_results"     # PowerShell: $env:TRACE_INSTRUMENTS_RESULTS_DIR = "$PWD/rerun_results"
python -m analysis.paper_numbers                              # numbers from the rerun ...
python -m plotting.make_figures                               # ... and figures
```

Without the environment variable the analysis and the figures read the shipped `results/`. Each results
directory has a `runs_manifest.json` that records which invocation produced each file and a
`runs_checksums.json` with the SHA-256 of each file as written. A run is skipped only when its files exist,
come from the same invocation and still have the recorded checksum; the runner never overwrites files of
another invocation, or files changed since, unless `--force` is given. The manifest does not record the
version of the code, so after changing code or dependencies use `--force` or a new `--results-dir`. The shipped `results/` therefore cannot be replaced by accident, and a replay
belongs in its own `--results-dir`. Replaying everything takes about 60 hours on the reference machine if the runs are made one after the
other; they are independent and can be distributed. The largest are the prediction tasks (13 h for the tanh
cell and 9 h for the gated cell, on the GPU), the cost frontier (9 h on the CPU) and the gated stationary-risk
run (4.5 h); all others take between seconds and two hours.

**What to expect from a rerun.** Every random draw is seeded, and a run repeated on the same machine, device
and thread count gives the same file. Across devices, thread counts and library versions the floating-point
rounding differs, and learning for 10^5 steps amplifies it: individual trajectories then differ in late
digits, while medians, ranks, counts and the conclusions drawn from them are stable.
`python -m experiments.restoration.step_cost_benchmark` measures wall time and depends on the hardware.

### Headline numbers

Printed by `python -m analysis.paper_numbers` from the shipped result files:

| Quantity (paper) | Value |
|---|---|
| Stationary excess risk, measured / predicted, tanh cell, median over systems (Fig. 6b) | RTRL 1.18, SnAp-1 1.10, immediate 1.08, leak 0.5 1.12 |
| Same for the gated cell, plain constant step (Table 2) | RTRL 1.02, SnAp-1 1.02, immediate 0.99, leak 0.5 1.01 |
| Rank correlation of efficiency loss with gradient cosine (Fig. 6d) | −0.978 |
| Read-out of 3 of 6 units: rank of the SnAp-1 instrument (Table 1) | 24 of 48 |
| … excess risk of RTRL / SnAp-1 / SnAp-1 with restoration (Table 1) | 9.5e-06 / 6.7e-04 / 6.2e-06 |
| Misspecification: predicted and measured displacement, cosine above 0.95 (Fig. 7b) | 78 of 96 runs |
| SnAp-1 on a non-normal linear system: runs that leave the calibrated point (Fig. 5) | 11 of 16 (system 1); 0 of 16 with restoration at 3× the threshold |
| Prediction tasks, tanh cell, held-out MSE of SnAp-1 / RTRL from a random start (Table 3) | 59.4, 1.77, 5.63, 3.82, 0.96 |

## Map from the paper to the code

`python -m experiments.registry` prints the same mapping; `experiments/registry.py` is its source.
The last column gives the identifier each experiment carried during development, which appears in older notes.

| Paper | Experiment | Entry point (`python -m …`) | Result files (`results/…`) | Was |
|---|---|---|---|---|
| §9.1, Fig. 4a–c | Closed forms for linear dynamics; non-normality sweep | `experiments.linear_dynamics.closed_form_checks` | `linear_dynamics/closed_forms.json` | E1 |
| §9.1, Fig. 4d | Exact 3×3 system with a negative eigenvalue of the SnAp-1 cross-Gram | `experiments.linear_dynamics.snap1_exact_instability` | `linear_dynamics/snap1_exact_instability.json` | E1c |
| §9.1 and §9.6, Fig. 5 | SnAp-1 leaves a calibrated point; restoration threshold | `experiments.restoration.stability_threshold` | `restoration/stability_threshold/` | E5a |
| §9.2, Fig. 6a,b,d | Stationary risk against the Lyapunov prediction | `experiments.calibrated_point.stationary_risk` | `calibrated_point/stationary_risk/tanh_*` | E2 |
| §9.2, Fig. 6c | Asymptotic constant of the averaged iterate | `experiments.calibrated_point.efficiency_constant` | `calibrated_point/efficiency_constant/` | E2b |
| §9.3, Table 1 | Rank and risk under a sparse read-out | `experiments.relevance.readout_rank` | `relevance/readout_rank/tanh_*` | E3 |
| §9.4, Table 2 | Gated cells: stationary risk | `experiments.calibrated_point.stationary_risk --cell gru_reset_after --preconditioner identity` | `calibrated_point/stationary_risk/gru_*` | E2g |
| §9.4, Table 2 | Gated cells: read-out rank | `experiments.relevance.readout_rank --cell gru_reset_after` (and `gru_reset_before`) | `relevance/readout_rank/gru_*` | E3g |
| §9.5, Fig. 7 | Displacement of the limit under misspecification | `experiments.misspecification.displacement` | `misspecification/displacement/` | E4 |
| §9.6 | Exchange rate of tangent probing | `experiments.restoration.exchange_rate` | `restoration/exchange_rate/` | E5b |
| §9.6, Fig. 8a–c | Risk against arithmetic for restoration by probes | `experiments.restoration.cost_frontier` | `restoration/cost_frontier/` | E8 |
| §9.6, Fig. 8d | Time per step of the recursions | `experiments.restoration.step_cost_benchmark` | `restoration/step_cost/` | bench |
| §9.7 | Distance from the calibrated point | `experiments.calibrated_point.local_regime` | `calibrated_point/local_regime/` | E9 |
| §9.8, Table 3, Fig. 9 | Five prediction tasks, from scratch and after handover | `experiments.prediction_tasks.online_prediction` | `prediction_tasks/` | E7 |
| §9.8, Table 4 | A real series with disjoint data per seed (household power) | `experiments.prediction_tasks.online_prediction --tasks power` | `prediction_tasks/*_power.json` | added in revision |
| §9.9, Table 5 | Lag behind a drifting teacher | `experiments.nonstationarity.tracking_lag` | `nonstationarity/tracking_lag/` | added in revision |

Figures and tables:

| Paper | Produced by | File |
|---|---|---|
| Fig. 4 | `plotting.nonnormality` | `figures/nonnormal.pdf` |
| Fig. 5 | `plotting.stability_threshold` | `figures/stabilise.pdf` |
| Fig. 6 | `plotting.calibrated_point` | `figures/wellspec.pdf` |
| Fig. 7 | `plotting.misspecification` | `figures/misspec.pdf` |
| Fig. 8 | `plotting.cost_frontier` | `figures/frontier.pdf` |
| Fig. 9 | `plotting.prediction_tasks` | `figures/tasks.pdf` |
| Table 1 | `python -m analysis.paper_numbers relevance` | printed |
| Table 2 | `python -m analysis.paper_numbers calibrated_point relevance` | printed |
| Table 3 | `python -m analysis.paper_numbers prediction_tasks` | printed, with LaTeX rows |
| Table 4 | `python -m analysis.paper_numbers prediction_tasks` | printed (last blocks) |
| Table 5 | `python -m analysis.paper_numbers nonstationarity` | printed |

Figs. 1–3 of the paper are schematic drawings and are not generated by code.

## Repository structure

```
trace_instruments/          the library (no experiment-specific code)
    cells.py                tanh / linear / GRU cells with analytic state and parameter Jacobians
    rules.py                names of the update rules and their parameters
    traces.py               trace recursions: RTRL, SnAp-1, immediate, leak, TBPTT-K, hybrid
    restoration.py          online estimators of the bias field (exact, projected, EMA, UORO)
    teacher_student.py      teacher-student problems, families of recurrent matrices
    stationary_moments.py   cross-Gram, instrument Gram, coherence spectrum, Lyapunov prediction
    learning.py             lane-batched online learning and paired risk evaluation
    linear_theory.py        exact moments for linear dynamics
    prediction_streams.py   delay, NARMA-10, Mackey-Glass, Lorenz, sunspot and household-power streams
    online_learner.py       recurrent learner with a trained read-out (prediction tasks)
    statistics.py           bootstrap interval, Wilcoxon test, Holm correction
    paths.py, config.py     locations of data / results / figures, precision, default device
experiments/
    linear_dynamics/  calibrated_point/  relevance/  misspecification/  restoration/  prediction_tasks/
    nonstationarity/
    registry.py             experiment <-> paper section, figure, table, result files
    reproduce.py            replays configs/paper_runs.json (or the demo)
    common.py               command line and result-file helpers
analysis/                   one module per section of the paper; paper_numbers.py runs all of them
plotting/                   style.py and one module per figure; make_figures.py draws all of them
configs/                    paper_runs.json (exact runs of the paper), demo_runs.json
results/                    result files of the paper, in the layout the experiments write
data/                       sunspot.txt (monthly sunspot numbers), household_power_segments.txt (twenty
                            disjoint segments of a household power record) and the script that made it
tests/                      unit tests and regression tests with reference outputs
```

**Paths and configuration.** All locations are resolved in `trace_instruments/paths.py`. Results go to
`results/` and figures to `figures/` unless `TRACE_INSTRUMENTS_RESULTS_DIR` / `TRACE_INSTRUMENTS_FIGURES_DIR`
are set; every experiment also accepts `--output-dir`, and all but the two that do not use `torch`
(`closed_form_checks`, `snap1_exact_instability`) accept `--device`, which defaults to the first GPU when
there is one and to the CPU otherwise (the stability-threshold and exchange-rate experiments default to the
CPU). `--help` documents the arguments. The defaults of an entry point are convenient sizes, not
necessarily those of the paper: the settings of the paper are the ones spelled out, argument by argument,
in `configs/paper_runs.json`.

## Using the library

Rules are named by strings: `rtrl`, `snap1`, `immediate`, `leak:<rate>`, `uoro`, `tbptt:<K>` and
`hybrid:<m>` (SnAp-1 for units `0..m-1`, exact sensitivity for the rest). A *lane* is one independent run;
all computations are batched over lanes. The cross-Gram of a rule at the teacher's parameters:

```python
from trace_instruments.teacher_student import TeacherStudentSystem
from trace_instruments.stationary_moments import estimate_stationary_moments, spectral_summary

system = TeacherStudentSystem(teacher_units=6, student_units=6, input_dim=2, num_systems=4, device="cpu")
moments = estimate_stationary_moments(system, ["snap1", "immediate"], num_steps=20_000)
C, Sigma, G = moments.cross_gram["snap1"], moments.instrument_gram["snap1"], moments.cross_gram["rtrl"]
for summary in spectral_summary(C, Sigma, G):       # one per system
    print(summary["min_real_eig"], summary["efficiency_loss"], summary["rank"])
```

A new cell implements the contract of `cells.RecurrentCell` (a step that returns the state Jacobian and the
immediate Jacobian); a new deterministic trace rule is a propagation operator in `traces.TraceBank.step`.

## Result files

Every experiment writes JSON with the same conventions: `args` holds the resolved command line (the exact
rational example takes no arguments and stores none), arrays are indexed by system (or seed) first, and names are spelled out (`excess_risk`, `gradient_cosine`, `num_parameters`, …).
Diverged runs are flagged in `diverged` and stored as `NaN`; the analysis counts them and leaves them out of
ratios. The module docstring of each experiment describes its lanes and outputs.

**Devices and the manifest.** `experiments.reproduce` records every result file it writes in
`runs_manifest.json` with the module, the arguments and the kind of device actually used. A run that the
paper made on a GPU and that falls back to the CPU is recorded as a CPU run, so it is not mistaken later
for the GPU run (and the shipped GPU results are not skipped-as-equal on a machine without a GPU: the
runner refuses to overwrite them unless `--results-dir` points elsewhere or `--force` is given). `--list`
prints the commands as this machine would run them (with `--device cpu` where no GPU is found) and, when
`--results-dir` is given, first the `export` line without which a copied command would write to `results/`. The two real series (`sunspot`, `power`) are fixed records
split 80/20; `--train-length` and `--eval-length` apply to the synthetic tasks only and are stored in
the result files of the real series without effect (2662/666 steps for `sunspot`, 6399/1600 for `power`).

## Tests

```bash
python -m pytest            # about 3 minutes on a CPU
```

- `test_cells.py`, `test_traces.py` — analytic Jacobians and the RTRL trace against automatic
  differentiation; limiting cases of the rules.
- `test_linear_theory.py`, `test_statistics.py` — closed-form identities and the statistics of the tables.
- `test_reference_runs.py` — eleven of the thirteen entry points at a tiny size against reference outputs
  stored in `tests/reference/`, and the exact rational-arithmetic example against the shipped result. The
  references were written by the code as it was when the paper's results were produced, except the two
  cases added in revision (`tracking_lag`, `online_prediction_power`), which this code wrote. The tolerance (relative `1e-6`)
  allows for other platforms and library versions; with `TRACE_INSTRUMENTS_EXACT=1` it is zero, which
  holds on the reference platform. The closed-form sweep (minutes) and the wall-time benchmark have no
  reference run and are exercised by the demo.
- `test_configs.py` — every invocation in `configs/` is accepted by its entry point, and every shipped
  result file stores exactly the arguments of the invocation that `configs/paper_runs.json` lists for it.

## Data, citation and licence

`data/sunspot.txt` holds the monthly mean total sunspot number, version 2.0, from WDC-SILSO, Royal
Observatory of Belgium, Brussels (<https://www.sidc.be/SILSO/>), which distributes it under CC BY-NC 4.0.

`data/household_power_segments.txt` is derived from the *Individual Household Electric Power Consumption*
data set of G. Hébrail and A. Bérard (UCI Machine Learning Repository, 2006,
<https://doi.org/10.24432/C58K54>, CC BY 4.0): the global active power averaged over ten-minute bins, cut
into twenty disjoint segments of 8000 bins without missing values. `data/prepare_household_power.py`
rebuilds the file from the original download, and `data/check_household_power_split.py` confirms that the
rebuilt file equals the shipped one and that no interpolated gap crosses a training/held-out split (its
output is kept in `data/household_power_split_check.txt`). Seed `s` of the `power` task uses segment `s`.

All other data are generated by the code from fixed seeds.

If you use this code, please cite the paper:

```bibtex
@article{credit_assignment_instruments,
  title  = {Credit-assignment rules as instruments: what an approximate online learning rule converges to, and when},
  author = {Yi, Junfei and Wang, Yuxiang and Ma, Chengyu},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

Licence: to be stated in `LICENSE` before release.
