"""Section 9.8: prediction tasks with a learned read-out (Table 3).

Statistics.  The unit of replication is the seed (initialisation and, for synthetic streams, the data).
Comparisons with RTRL are paired: for each seed the ratio of the two errors.  Reported are the median
ratio, a 95% bootstrap interval for the median (10 000 resamples), and a two-sided Wilcoxon signed-rank
test of log ratio = 0 with Holm correction over all the comparisons of one table.  Diverged runs are
counted and excluded from the ratios.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from analysis.common import load_optional, result_files
from trace_instruments import cells
from trace_instruments.prediction_streams import POWER, TASKS, make_series, task_tag
from trace_instruments.statistics import bootstrap_median_interval, holm_adjust, wilcoxon_log_ratio

RESULTS_SUBDIR = "prediction_tasks"
TASK_TAGS = [task_tag(task) for task in TASKS]
# The long real series, with disjoint data per seed, is reported apart from the five tasks of the task
# table and has its own family of tests.
DISJOINT_DATA_TAGS = [task_tag(POWER)]
# (cell, number of units) of the runs reported in the paper, in the order of the report
REPORTED_RUNS = [
    (cells.TANH, 16),
    (cells.GRU_RESET_AFTER, 16),
    (cells.TANH, 8),
    (cells.TANH, 32),
    (cells.TANH, 64),
    (cells.GRU_RESET_AFTER, 32),
]
TABLE_CELLS = (cells.TANH, cells.GRU_RESET_AFTER)
TABLE_NUM_UNITS = 16
LATEX_RULE_NAMES = {
    "snap1": "SnAp-1",
    "immediate": "immediate",
    "leak:0.5": "leak 0.5",
    "uoro": "UORO",
    "tbptt:8": "TBPTT-8",
}
TRACE_RULE_INDICES = (1, 2, 3)  # snap1, immediate, leak


def load_run(cell, num_units, tag):
    return load_optional(RESULTS_SUBDIR, f"{cell}_n{num_units}_{tag}.json")


def select_scratch_steps(d):
    """Step per rule: smallest median online loss over the last tenth, diverged runs counted as infinite."""
    online = np.array(d["online_loss_scratch"], float)
    mse = np.array(d["scratch"]["mse"], float)
    online = np.where(np.isfinite(mse), online, np.inf)
    return np.argmin(np.median(online, 0), 0)


def select_handover_steps(d):
    """Handover step per rule: smallest median online loss over the last tenth of the handover phase."""
    online = np.array(d["online_loss_handover"], float)
    mse = np.array(d["handover"]["mse"], float)
    return np.argmin(np.median(np.where(np.isfinite(mse), online, np.inf), 0), 0)


def handover_ratios(d, r):
    """Per-seed ratio of the error of rule ``r`` to that of RTRL after handover.

    Returns the ratio with each rule at its own step, the ratio at RTRL's step, and the rule's own step.
    """
    mse = np.array(d["handover"]["mse"], float)
    steps = select_handover_steps(d)
    a0 = d["handover_step_index"]
    return mse[:, steps[r], r] / mse[:, steps[0], 0], mse[:, a0, r] / mse[:, a0, 0], d["step_sizes"][steps[r]]


def frozen_error(d):
    """Per-seed error of the handover point itself (RTRL's end point, no further recurrent update)."""
    return np.array(d["handover_point"]["mse"], float)[:, d["handover_step_index"], 0]


def report_tasks(cell, num_units, task_tags=TASK_TAGS, title="Prediction tasks"):
    rows = []
    p_values = []
    print(f"\n== {title}: cell={cell} n={num_units} ==")
    for tag in task_tags:
        d = load_run(cell, num_units, tag)
        if d is None:
            continue
        rule_names, step_sizes = d["rules"], d["step_sizes"]
        selected = select_scratch_steps(d)
        mse = np.array(d["scratch"]["mse"], float)
        base = mse[:, selected[0], 0]
        predictability = np.array(d["scratch"]["score_predictability"])[:, selected[0], 0]
        print(
            f"-- {tag}: P={d['num_parameters']} seeds={mse.shape[0]}  RTRL step {step_sizes[selected[0]]:g}: "
            f"median MSE {np.nanmedian(base):.4g} "
            f"IQR [{np.nanpercentile(base, 25):.3g}, {np.nanpercentile(base, 75):.3g}] "
            f"diverged {int(np.isnan(base).sum())}"
            f" | predictable share of the residual (R^2 on ten lags) at RTRL's end point: "
            f"median {np.nanmedian(predictability):.3f}"
        )
        handover_mse = np.array(d["handover"]["mse"], float)
        handover_steps = select_handover_steps(d)
        frozen = frozen_error(d)
        print(
            f"   handover from RTRL at step {step_sizes[d['handover_step_index']]:g}; "
            f"RTRL continues at step {step_sizes[handover_steps[0]]:g}; frozen end point / RTRL continued: "
            f"median {np.nanmedian(frozen / handover_mse[:, handover_steps[0], 0]):.3g}; "
            + "better than frozen in (of %d seeds): " % len(frozen)
            + ", ".join(
                f"{rule_names[r]} {int(np.nansum(handover_mse[:, handover_steps[r], r] < frozen))}"
                for r in range(len(rule_names))
            )
        )
        for r in range(1, len(rule_names)):
            x = mse[:, selected[r], r]
            ratio = x / base
            own, at_rtrl_step, own_step = handover_ratios(d, r)
            rows.append((tag, rule_names[r], step_sizes[selected[r]], x, ratio, own, at_rtrl_step, own_step))
            p_values += [wilcoxon_log_ratio(ratio), wilcoxon_log_ratio(own)]
    adjusted = holm_adjust(p_values)
    for i, (tag, rule, step, x, ratio, own, at_rtrl_step, own_step) in enumerate(rows):
        low, high = bootstrap_median_interval(ratio)
        own_low, own_high = bootstrap_median_interval(own)
        print(
            f"   {tag:8s} {rule:9s} step {step:<7g} MSE {np.nanmedian(x):.4g} | "
            f"scratch ratio to RTRL {np.nanmedian(ratio):8.3g} [{low:.3g}, {high:.3g}] "
            f"worse in {int(np.nansum(ratio > 1))}/{int(np.isfinite(ratio).sum())} p_holm={adjusted[2 * i]:.2g} "
            f"div={int(np.isnan(x).sum())} "
            f"| handover (own step {own_step:g}) ratio {np.nanmedian(own):6.3g} [{own_low:.3g}, {own_high:.3g}] "
            f"worse in {int(np.nansum(own > 1))}/{int(np.isfinite(own).sum())} p_holm={adjusted[2 * i + 1]:.2g} "
            f"div={int(np.isnan(own).sum())}"
            f" ; at RTRL's step {np.nanmedian(at_rtrl_step):.3g} div={int(np.isnan(at_rtrl_step).sum())}"
        )
    # what predicts the handover loss: the rule's own mean update and bias field at the handover point
    predictors = {"field/grad": [], "bias/grad": [], "r2": []}
    log_ratios = []
    for tag in task_tags:
        d = load_run(cell, num_units, tag)
        if d is None:
            continue
        a0 = d["handover_step_index"]
        point = {key: np.array(value, float) for key, value in d["handover_point"].items()}
        for r in TRACE_RULE_INDICES:
            log_ratios += list(np.log(handover_ratios(d, r)[0]))
            gradient = point["gradient_norm"][:, a0, r]
            predictors["field/grad"] += list(point["mean_update_norm"][:, a0, r] / gradient)
            predictors["bias/grad"] += list(point["bias_field_norm"][:, a0, r] / gradient)
            predictors["r2"] += list(point["score_predictability"][:, a0, r])
    log_ratios = np.array(log_ratios)
    for name, values in predictors.items():
        values = np.array(values)
        ok = np.isfinite(values) & np.isfinite(log_ratios)
        if ok.sum() > 10:
            s = spearmanr(values[ok], log_ratios[ok])
            print(
                f"   Spearman(log handover ratio, {name} at the handover point) = {s.statistic:+.2f} "
                f"(n={int(ok.sum())}, p={s.pvalue:.1g})"
            )


def format_ratio(value):
    if not np.isfinite(value):
        return "--"
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def report_latex_rows():
    """Rows of the task table: median paired ratio to RTRL; a star marks Holm-adjusted p < 0.05 per cell."""
    print("\n== LaTeX rows (scratch | handover), tasks in the order " + ", ".join(TASK_TAGS) + " ==")
    for cell in TABLE_CELLS:
        records = [load_run(cell, TABLE_NUM_UNITS, tag) for tag in TASK_TAGS]
        if any(d is None for d in records):
            continue
        rule_names = records[0]["rules"]
        scratch, handover = {}, {}
        for ti, d in enumerate(records):
            selected = select_scratch_steps(d)
            mse = np.array(d["scratch"]["mse"], float)
            for r in range(1, len(rule_names)):
                scratch[(r, ti)] = mse[:, selected[r], r] / mse[:, selected[0], 0]
                handover[(r, ti)] = handover_ratios(d, r)[0]
        keys = [(r, ti) for ti in range(len(TASK_TAGS)) for r in range(1, len(rule_names))]
        adjusted = holm_adjust(
            [wilcoxon_log_ratio(scratch[k]) for k in keys] + [wilcoxon_log_ratio(handover[k]) for k in keys]
        )
        significant = {("S",) + k: adjusted[i] < 0.05 for i, k in enumerate(keys)}
        significant.update({("H",) + k: adjusted[len(keys) + i] < 0.05 for i, k in enumerate(keys)})
        rtrl_mse = [
            np.nanmedian(np.array(d["scratch"]["mse"], float)[:, select_scratch_steps(d)[0], 0]) for d in records
        ]
        print(f"% {cell}: RTRL median MSE " + " & ".join(f"{value:.3g}" for value in rtrl_mse))
        frozen_ratio = [
            np.nanmedian(frozen_error(d) / np.array(d["handover"]["mse"], float)[:, select_handover_steps(d)[0], 0])
            for d in records
        ]
        print(f"{cell} H frozen     & " + " & ".join(format_ratio(value) for value in frozen_ratio) + r" \\")
        for block, ratios in (("S", scratch), ("H", handover)):
            for r in range(1, len(rule_names)):
                entries = []
                for ti in range(len(TASK_TAGS)):
                    x = ratios[(r, ti)]
                    num_diverged = int(np.isnan(x).sum())
                    if num_diverged > len(x) // 2:
                        entry = "--"
                    else:
                        entry = format_ratio(np.nanmedian(x)) + ("$^*$" if significant[(block, r, ti)] else "")
                    entries.append(entry + (f"\\,({num_diverged})" if num_diverged else ""))
                print(f"{cell} {block} {LATEX_RULE_NAMES[rule_names[r]]:10s} & " + " & ".join(entries) + r" \\")


def report_linear_baselines(task, num_seeds=20):
    """Held-out error of two predictors without memory, for comparison with the networks: the last value
    repeated, and one coefficient fitted to the training series by least squares."""
    persistence, one_coefficient = [], []
    for seed in range(num_seeds):
        u_train, y_train, u_eval, y_eval = make_series(task, seed, 0, 0)
        coefficient = np.dot(u_train, y_train) / np.dot(u_train, u_train)
        persistence.append(np.mean((y_eval - u_eval) ** 2))
        one_coefficient.append(np.mean((y_eval - coefficient * u_eval) ** 2))
    print(
        f"\n== Baselines without memory, {task}: held-out MSE, median over {num_seeds} seeds: "
        f"last value repeated {np.median(persistence):.3f}, one fitted coefficient {np.median(one_coefficient):.3f} =="
    )


def report():
    for cell, num_units in REPORTED_RUNS:
        if result_files(RESULTS_SUBDIR, pattern=f"{cell}_n{num_units}_*.json"):
            report_tasks(cell, num_units)
    report_latex_rows()
    report_linear_baselines(POWER)
    for cell in TABLE_CELLS:
        if all(load_run(cell, TABLE_NUM_UNITS, tag) is not None for tag in DISJOINT_DATA_TAGS):
            report_tasks(cell, TABLE_NUM_UNITS, DISJOINT_DATA_TAGS, "Real series with disjoint data per seed")


if __name__ == "__main__":
    report()
