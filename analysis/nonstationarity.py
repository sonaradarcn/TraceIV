"""Section 9.9: a drifting teacher (Table 5).

For every rule and step size: the lag behind the moving calibrated point against the prediction of the
linearised recursion (cosine and ratio of norms), the lag relative to the norm of the parameters, and the
cost of the drift in excess risk against its prediction.  Medians over systems, with the range.
"""

from __future__ import annotations

import numpy as np

from analysis.common import load, result_files

RESULTS_SUBDIR = ("nonstationarity", "tracking_lag")


def spread(values):
    values = np.array(values, float)
    return f"{np.nanmedian(values):.3g} [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]"


def report():
    for path in result_files(*RESULTS_SUBDIR):
        d, name = load(path), path.stem
        args = d["args"]
        readout = "all units read" if args["num_readout"] is None else f"{args['num_readout']} units read"
        print(
            f"\n== Drifting teacher: {name} ({readout}; P={d['num_parameters']}, {args['num_systems']} systems, "
            f"drift {args['total_drift']} |theta*| over {args['learning_steps']} steps, noise {args['noise_std']}) =="
        )
        for rule in d["rules"]:
            result = d["results"][rule]
            for step_size in d["step_sizes"]:
                rows = result["step_sizes"][str(step_size)]
                cost_ratio = np.array(rows["measured_cost"], float) / np.array(rows["predicted_cost"], float)
                print(
                    f"  {rule:9s} step {step_size:<6g} rank {int(np.median(result['rank']))}/{d['num_parameters']} | "
                    f"lag cosine {spread(rows['lag_cosine'])} | norm measured/predicted {spread(rows['lag_norm_ratio'])} | "
                    f"lag/|theta*| {spread(rows['measured_lag'])} | "
                    f"cost measured {np.nanmedian(rows['measured_cost']):.3g} predicted "
                    f"{np.nanmedian(rows['predicted_cost']):.3g} ratio {spread(cost_ratio)} | "
                    f"diverged {int(np.sum(rows['diverged']))} | "
                    f"spectral radius of the mean recursion, largest over systems {np.max(rows['spectral_radius']):.6f}"
                )
        reference = d["results"][d["rules"][0]]["step_sizes"]
        for step_size in d["step_sizes"]:
            base = np.array(reference[str(step_size)]["measured_cost"], float)
            ratios = ", ".join(
                f"{rule} {np.nanmedian(np.array(d['results'][rule]['step_sizes'][str(step_size)]['measured_cost'], float) / base):.3g}"
                for rule in d["rules"][1:]
            )
            print(f"  cost relative to {d['rules'][0]} at step {step_size:g} (median of per-system ratios): {ratios}")


if __name__ == "__main__":
    report()
