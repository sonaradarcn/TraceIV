"""Sections 9.3 and 9.4: rank of the instrument and excess risk under a sparse read-out."""

from __future__ import annotations

import numpy as np
from scipy import stats

from analysis.common import load_all
from trace_instruments import cells, rules
from trace_instruments.restoration import EXACT
from trace_instruments.statistics import wilcoxon_log_ratio

READOUT_RANK = ("relevance", "readout_rank")


def report_tanh():
    print("== Relevance: read-out rank (tanh)")
    records = [
        d for d in load_all(*READOUT_RANK) if d["args"]["cell"] == cells.TANH and d["args"]["restoration"] == EXACT
    ]
    for readout in ("units", "dense"):
        selected = [d for d in records if d["args"]["readout"] == readout]
        for d in sorted(selected, key=lambda d: -d["num_readout"]):
            excess = np.where(np.array(d["diverged"]), np.nan, np.array(d["excess_risk"]))
            num_systems = excess.shape[0]
            print(
                f"  {readout} m={d['num_readout']} rank {int(np.median(d['rank'][rules.SNAP1]))}/{d['num_parameters']} "
                f"rho2min {np.median(np.min(np.array(d['coherence'][rules.SNAP1]), 1)):.3f} | "
                + " ".join(f"{name}={np.nanmedian(excess[:, j]):.2e}" for j, name in enumerate(d["lanes"]))
                + f" | start {np.median(d['excess_risk_start']):.2e} div {np.array(d['diverged']).sum(0).tolist()}"
            )
            if readout == "units" and d["num_readout"] < d["args"]["num_units"]:
                test = stats.wilcoxon(excess[:, 1], excess[:, 0])
                print(
                    f"       snap1 vs rtrl: paired Wilcoxon p={test.pvalue:.1e}; "
                    f"snap1 worse in {int((excess[:, 1] > excess[:, 0]).sum())}/{num_systems}; "
                    f"snap1+restoration vs rtrl worse in {int((excess[:, 3] > excess[:, 0]).sum())}/{num_systems}"
                )


def predicted_relevant_dimension(cell: str, n: int, m: int, fan_in: int) -> int:
    """Number of parameters a trace rule can reach when ``m`` of ``n`` units are read out."""
    if cell == cells.GRU_RESET_AFTER:
        return 3 * m * fan_in
    return (2 * m + n) * fan_in


def report_gated():
    print("\n== Relevance: read-out rank in gated cells ==")
    for d in load_all(*READOUT_RANK):
        args = d["args"]
        if args["cell"] not in cells.GATED_CELLS:
            continue
        n, m = args["num_units"], d["num_readout"]
        predicted = predicted_relevant_dimension(args["cell"], n, m, n + args["input_dim"])
        excess = np.array(d["excess_risk"], float)
        S = excess.shape[0]
        print(
            f"{args['cell']:16s} m={m} P={d['num_parameters']} predicted |P_O|={min(predicted, d['num_parameters'])}  "
            f"rank snap1 {sorted(set(d['rank'][rules.SNAP1]))} immediate {sorted(set(d['rank'][rules.IMMEDIATE]))} "
            f"rtrl {sorted(set(d['rank'][rules.RTRL]))} | "
            + "  ".join(f"{name}: {np.nanmedian(excess[:, i]):.3g}" for i, name in enumerate(d["lanes"]))
            + f" | start {np.median(d['excess_risk_start']):.3g} "
            f"| trace rule worse than RTRL in {int(np.nansum(excess[:, 1] > excess[:, 0]))}/{S} (snap1) "
            f"{int(np.nansum(excess[:, 2] > excess[:, 0]))}/{S} (immediate), "
            f"Wilcoxon p={wilcoxon_log_ratio(excess[:, 1] / excess[:, 0]):.2g}/"
            f"{wilcoxon_log_ratio(excess[:, 2] / excess[:, 0]):.2g} div={int(np.array(d['diverged']).sum())}"
        )


def report():
    report_tanh()
    report_gated()


if __name__ == "__main__":
    report()
