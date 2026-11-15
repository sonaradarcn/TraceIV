"""Figure ``misspec`` (Section 9.5): displacement of the limit and risk gap against the first-order prediction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.common import load_all
from plotting.style import COLOUR, DASHED, INK, LEAK_05, MARKER, MUTED, TEXT_WIDTH, PanelLetters, median_band
from plotting.style import no_minor_ticks, rule_handle, save, top_legend
from trace_instruments import rules

NAME = "misspec"
RULES = (rules.SNAP1, rules.IMMEDIATE, LEAK_05)


def draw():
    records = sorted(load_all("misspecification", "displacement"), key=lambda d: d["coupling"])
    couplings = [d["coupling"] for d in records]
    fig, ax = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.15))
    cosines = np.array([[d["rules"][rule]["direction_cosine"] for rule in RULES] for d in records])
    # the run with the worst direction is ringed in panels a-c
    worst_coupling, worst_rule, worst_system = np.unravel_index(np.nanargmin(cosines), cosines.shape)
    ring = dict(s=38, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=5)
    for i, rule in enumerate(RULES):
        for j, d in enumerate(records):
            r = d["rules"][rule]
            ok = ~np.array(r["diverged"])
            x = j + (i - 1) * 0.24
            if (j, i) == (worst_coupling, worst_rule):
                s = worst_system
                ax[0].scatter([r["predicted_displacement"][s]], [r["displacement"][s]], **ring)
                ax[2].scatter([r["predicted_risk_gap"][s]], [r["risk_gap"][s]], **ring)
                ax[1].scatter([x], [r["direction_cosine"][s]], **ring)
            marks = dict(color=COLOUR[rule], marker=MARKER[rule], linewidths=0)
            ax[0].scatter(
                np.array(r["predicted_displacement"])[ok], np.array(r["displacement"])[ok], s=8, alpha=0.8, **marks
            )
            ax[2].scatter(np.array(r["predicted_risk_gap"])[ok], np.array(r["risk_gap"])[ok], s=8, alpha=0.8, **marks)
            cosine = np.array(r["direction_cosine"])[ok]
            ax[1].scatter(np.full(cosine.size, x), cosine, s=6, alpha=0.55, **marks)
            ax[1].plot([x - 0.09, x + 0.09], [np.median(cosine)] * 2, color=COLOUR[rule], lw=1.6, solid_capstyle="butt")
        gaps = [
            np.where(np.array(d["rules"][rule]["diverged"]), np.nan, np.array(d["rules"][rule]["risk_gap"]))
            for d in records
        ]
        median_band(ax[3], couplings, gaps, rule)
    for a, x_label, y_label in (
        (ax[0], r"predicted  $\|\tilde C^{-1}v\|$", r"measured  $\|\theta_Q-\theta_0\|$"),
        (ax[2], "predicted risk gap", "measured risk gap"),
    ):
        a.set_xscale("log")
        a.set_yscale("log")
        low = min(a.get_xlim()[0], a.get_ylim()[0])
        high = max(a.get_xlim()[1], a.get_ylim()[1])
        low, high = low / 1.3, high * 1.3
        a.plot([low, high], [low, high], color=MUTED, lw=0.8, zorder=0)
        a.set_xlim(low, high)
        a.set_ylim(low, high)
        a.set_xlabel(x_label)
        a.set_ylabel(y_label)
        a.set_aspect("equal", adjustable="box")
    ax[1].set_xticks(range(len(couplings)), [f"{e:g}" for e in couplings])
    ax[1].set_ylim(-0.12, 1.04)
    ax[1].axhline(1, color=MUTED, lw=0.6)
    ax[1].axhline(0, color=MUTED, lw=0.4)
    ax[1].set_xlabel(r"coupling  $\epsilon$")
    ax[1].set_ylabel("cosine(measured, predicted)")
    ends = np.array([couplings[0], couplings[-1]])
    anchor = np.median(records[0]["rules"][rules.IMMEDIATE]["risk_gap"]) * 0.3
    ax[3].plot(ends, anchor * (ends / ends[0]) ** 2, color=MUTED, lw=0.8, ls=DASHED)
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xticks(couplings, [f"{x:g}" for x in couplings])
    ax[3].set_xlabel(r"coupling  $\epsilon$")
    ax[3].set_ylabel(r"risk gap  $L(\theta_Q)-L(\theta_0)$")
    for a in ax:
        a.grid(True)
    no_minor_ticks(ax)
    PanelLetters()(ax)
    top_legend(
        fig,
        [rule_handle(rule, line=False) for rule in RULES]
        + [
            Line2D([], [], color=MUTED, lw=0.8, ls=DASHED, label=r"reference slope $\propto\epsilon^2$"),
            Line2D(
                [], [], color=INK, marker="o", ls="none", mfc="none", mew=0.8, markersize=6, label="same run in a–c"
            ),
        ],
    )
    save(fig, NAME)


if __name__ == "__main__":
    draw()
