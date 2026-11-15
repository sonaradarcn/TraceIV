"""Figure ``wellspec`` (Section 9.2).

Coherence spectrum (a), stationary excess risk (b) and asymptotic constant of the averaged iterate (c)
against their predictions, and efficiency loss against gradient cosine (d).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import load_by_family
from plotting.style import COLOUR, LEAK_05, LEAK_09, MARKER, MUTED, TEXT_WIDTH, PanelLetters, no_minor_ticks
from plotting.style import rule_handle, save, top_legend
from trace_instruments import rules

NAME = "wellspec"
RULES = (rules.RTRL, rules.SNAP1, rules.IMMEDIATE, LEAK_05, LEAK_09)


def draw():
    fig, ax = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.15))
    risk_records = load_by_family("calibrated_point", "stationary_risk", pattern="tanh_*.json")

    def points(axis, x, y, rule):
        axis.scatter(x, y, s=8, color=COLOUR[rule], marker=MARKER[rule], linewidths=0, alpha=0.8)

    # a: canonical correlations, sorted, pooled over systems
    for rule in RULES[1:]:
        correlations = np.array(
            [
                np.sqrt(np.clip(np.sort(np.array(row, float)), 0, 1))
                for d in risk_records
                for row in d["rules"][rule]["coherence"]
                if len(row) == d["num_parameters"]
            ]
        )
        x = (np.arange(correlations.shape[1]) + 0.5) / correlations.shape[1]
        ax[0].fill_between(
            x,
            np.nanpercentile(correlations, 25, 0),
            np.nanpercentile(correlations, 75, 0),
            color=COLOUR[rule],
            alpha=0.12,
            lw=0,
        )
        ax[0].plot(x, np.nanmedian(correlations, 0), color=COLOUR[rule])
    ax[0].axhline(1, color=COLOUR[rules.RTRL], lw=1.3)
    ax[0].set_xlabel(r"rank of direction  $i/P$")
    ax[0].set_ylabel(r"canonical correlation  $\rho_i$")
    ax[0].set_ylim(0, 1.06)
    ax[0].set_xlim(0, 1)

    # b: stationary excess risk; d: efficiency loss against gradient cosine
    for d in risk_records:
        for rule, r in d["rules"].items():
            ok = ~np.array(r["diverged"])
            points(ax[1], np.array(r["predicted_excess_risk"])[ok], np.array(r["measured_excess_risk"])[ok], rule)
            if rule != rules.RTRL:
                points(ax[3], np.array(r["gradient_cosine"])[ok], np.array(r["efficiency_loss"])[ok], rule)
    limits = [1e-3, 3e-2]
    ticks = [1e-3, 3e-3, 1e-2, 3e-2]
    tick_labels = ["0.001", "0.003", "0.01", "0.03"]
    ax[1].plot(limits, limits, color=MUTED, lw=0.8, zorder=0)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(limits)
    ax[1].set_ylim(limits)
    ax[1].set_xticks(ticks, tick_labels)
    ax[1].set_yticks(ticks, tick_labels)
    ax[1].set_xlabel("predicted excess risk")
    ax[1].set_ylabel("measured excess risk")

    # c: asymptotic constant of the averaged iterate
    for d in load_by_family("calibrated_point", "efficiency_constant"):
        P = d["num_parameters"]
        for rule, r in d["rules"].items():
            points(ax[2], np.array(r["predicted_constant"]) / P, np.array(r["measured_constant"]) / P, rule)
    limits = [0.9, 11]
    ticks = [1, 2, 4, 8]
    ax[2].plot(limits, limits, color=MUTED, lw=0.8, zorder=0)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlim(limits)
    ax[2].set_ylim(limits)
    ax[2].set_xticks(ticks, [str(t) for t in ticks])
    ax[2].set_yticks(ticks, [str(t) for t in ticks])
    ax[2].set_xlabel(r"predicted  $P^{-1}\sum_i\rho_i^{-2}$")
    ax[2].set_ylabel("measured asymptotic constant")

    ax[3].set_yscale("log")
    ax[3].set_ylim(0.9, 300)
    ax[3].set_xlim(0, 1.02)
    ax[3].set_yticks([1, 10, 100], ["1", "10", "100"])
    ax[3].set_xlabel("gradient cosine")
    ax[3].set_ylabel(r"efficiency loss  $P^{-1}\sum_i\rho_i^{-2}$")
    for a in ax:
        a.grid(True)
    no_minor_ticks(ax)
    PanelLetters()(ax)
    top_legend(fig, [rule_handle(rule, line=False) for rule in RULES])
    save(fig, NAME)


if __name__ == "__main__":
    draw()
