"""Section 9.6: stability threshold, exchange rate, cost frontier and step cost of restoration."""

from __future__ import annotations

import numpy as np

from analysis.common import load, load_all, result_files
from trace_instruments.statistics import bootstrap_median_interval


def report_stability_threshold():
    print("== Restoration: stability threshold")
    for d in load_all("restoration", "stability_threshold"):
        distance = np.array(d["distance"])
        print(
            f"  system {d['seed']}: minRe/lmax {d['min_real_eig'] / d['gauss_newton_max_eig']:+.4f} "
            f"thr beta {d['threshold_rate']:.2e} | "
            + " | ".join(
                f"{name}: final median {np.median(distance[:, j, -1]):.2e}, "
                f"escaped {(distance[:, j, -1] > 1).sum()}/{distance.shape[0]}"
                for j, name in enumerate(d["lanes"])
            )
        )


def report_exchange_rate():
    print("== Restoration: exchange-rate law (every configuration)")
    settled, transient = [], []
    for d in load_all("restoration", "exchange_rate"):
        for c in d["configurations"]:
            contraction = c["predicted_contraction"]
            measured, predicted = np.array(c["error_curve_measured"]), np.array(c["error_curve_predicted"])
            cover = len(measured) * (1 - contraction)  # run length in mean-square relaxation times
            j = max(2, int(min(len(measured), 1.0 / (1 - contraction))))
            deviation = float(np.median(np.abs(measured[1:j] / predicted[1:j] - 1)))
            ratio = float(np.median(c["stationary_error_measured"]) / np.median(c["stationary_error_predicted"]))
            print(
                f"  P={c['num_parameters']:3d} k={c['num_tangents']:2d} beta={c['tracking_rate']:.2f} "
                f"w={c['window']:3d}: relaxation times covered {cover:7.2f}; "
                f"decay deviation over first relaxation time {deviation:.3f}; "
                f"final error / predicted stationary {ratio:.2f}" + ("" if cover >= 4 else "  (transient)")
            )
            (settled if cover >= 4 else transient).append((ratio, deviation))
    if not settled or not transient:
        return
    a, t = np.array(settled), np.array(transient)
    inside = (a[:, 0] >= 0.86) & (a[:, 0] <= 1.18)
    print(
        f"  configurations covering >= 4 relaxation times: {len(a)}; ratio in [0.86,1.18]: {int(inside.sum())}; "
        f"others: {np.round(a[~inside, 0], 2).tolist()}"
    )
    print(
        f"  decay deviation, median over all {len(a) + len(t)} configurations: "
        f"{np.median(np.concatenate([a[:, 1], t[:, 1]])):.3f}; over the {len(t)} transient ones: {np.median(t[:, 1]):.3f}"
    )


def report_cost_frontier():
    print("\n== Restoration: cost (multiply-adds of the recursions per step, relative to RTRL) ==")
    for d in load_all("restoration", "cost_frontier"):
        excess = np.array(d["excess_risk"], float)
        cost = np.array(d["arithmetic_cost"], float) / d["arithmetic_cost"][0]
        lanes = d["lanes"]
        median = np.nanmedian(excess, 0)
        S = excess.shape[0]
        print(
            f"-- n={d['num_units']} m={d['num_readout']} P={d['num_parameters']} blind P_e={d['num_blind_parameters']}"
        )
        for j in range(3):
            low, high = bootstrap_median_interval(excess[:, j])
            print(
                f"   {lanes[j]['rule']:8s} cost {cost[j]:.4f} excess {median[j]:.3e} [{low:.2e}, {high:.2e}] "
                f"div={int(np.isnan(excess[:, j]).sum())}/{S}"
                + (f"  ratio to RTRL {np.nanmedian(excess[:, j] / excess[:, 0]):.2f}" if j else "")
            )
        for k in sorted({lane["probes"]["num_tangents"] for lane in lanes[3:]}):
            group = [j for j in range(3, len(lanes)) if lanes[j]["probes"]["num_tangents"] == k]

            def usable_median(j):
                return median[j] if np.isfinite(median[j]) and np.isnan(excess[:, j]).sum() == 0 else np.inf

            best = min(group, key=usable_median)
            low, high = bootstrap_median_interval(excess[:, best])
            probes = lanes[best]["probes"]
            print(
                f"   probes k={k:<4d} cost {cost[best]:.4f} best excess {median[best]:.3e} [{low:.2e}, {high:.2e}] "
                f"at beta={probes['tracking_rate']} w={probes['window']} "
                f"(ratio to RTRL {np.nanmedian(excess[:, best] / excess[:, 0]):.2f}, "
                f"to SnAp-1 {np.nanmedian(excess[:, best] / excess[:, 1]):.3f}); "
                f"worst setting {np.nanmax(median[group]):.3e}; "
                f"settings with a diverged run {sum(int(np.isnan(excess[:, j]).any()) for j in group)}/{len(group)}; "
                f"settings worse than SnAp-1 alone {sum(int(not median[j] < median[1]) for j in group)}/{len(group)}"
            )


def report_step_cost():
    print("\n== Restoration: per-step cost (one learner, float64) ==")
    for path in result_files("restoration", "step_cost"):
        d = load(path)
        print(f"-- {path.stem} ({d.get('hardware', '')})")
        sizes = sorted({row["num_units"] for row in d["rows"]})
        rule_names = list(dict.fromkeys(row["rule"] for row in d["rows"]))
        table = {(row["rule"], row["num_units"]): row for row in d["rows"]}

        def line(rule, key, fmt):
            return f"   {rule:9s} " + "".join(
                format(table[(rule, n)][key], fmt) if (rule, n) in table else " " * 11 for n in sizes
            )

        print("   us/step   " + "".join(f"n={n:<9d}" for n in sizes))
        for rule in rule_names:
            print(line(rule, "microseconds_per_step", "<11.0f"))
        print("   trace MiB")
        for rule in rule_names:
            print(line(rule, "trace_mebibytes", "<11.3g"))


def report():
    report_stability_threshold()
    report_exchange_rate()
    report_cost_frontier()
    report_step_cost()


if __name__ == "__main__":
    report()
