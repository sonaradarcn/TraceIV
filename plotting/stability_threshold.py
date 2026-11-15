"""Figure ``stabilise`` (Sections 9.1 and 9.6).

SnAp-1 leaves ``theta*`` while the immediate rule and RTRL stay; restoration holds it when the tracking
rate is above the threshold.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from analysis.common import load_all
from plotting.style import COLOUR, DASHED, MUTED, TEXT_WIDTH, PanelLetters, save, top_legend
from trace_instruments import rules

NAME = "stabilise"
ESCAPED_ABOVE = 4.99  # escaped lanes are frozen at the escape radius of the experiment, 5
RATE_MULTIPLES = ("0.3", "3", "10", "100")
# lanes drawn in the trajectory panels: (colour of rule, line style)
DRAWN_LANES = {
    rules.RTRL: (rules.RTRL, "-"),
    rules.IMMEDIATE: (rules.IMMEDIATE, "-"),
    rules.SNAP1: (rules.SNAP1, "-"),
    rules.restored_lane_name("0.3"): (rules.SNAP1, ":"),
    rules.restored_lane_name("10"): (rules.SNAP1, "--"),
}


def draw():
    records = sorted(load_all("restoration", "stability_threshold"), key=lambda d: d["seed"])
    fig, ax = plt.subplots(
        1, len(records) + 1, figsize=(TEXT_WIDTH, 2.15), gridspec_kw=dict(width_ratios=[1] * len(records) + [1.15])
    )
    for i, (a, d) in enumerate(zip(ax, records)):
        distance = np.array(d["distance"])
        x = np.linspace(0, 1, distance.shape[2] + 1)[1:] * d["args"]["num_steps"] / 1e3
        for j, lane in enumerate(d["lanes"]):
            if lane not in DRAWN_LANES:
                continue
            rule, line_style = DRAWN_LANES[lane]
            y = distance[:, j]
            a.fill_between(x, np.percentile(y, 25, 0), np.percentile(y, 75, 0), color=COLOUR[rule], alpha=0.15, lw=0)
            a.plot(x, np.median(y, 0), color=COLOUR[rule], ls=line_style)
        a.set_yscale("log")
        a.set_ylim(1e-5, 10)
        a.set_xlabel(r"steps ($\times10^3$)")
        a.set_title(f"system {i + 1}", loc="right", color=MUTED)
        a.grid(True, axis="y")
        a.set_xticks([0, 200, 400])
        a.set_yticks([1e-5, 1e-3, 1e-1, 10])
        if i:
            a.set_yticklabels([])
    ax[0].set_ylabel(r"distance  $\|\theta_t-\theta^\star\|$")

    # last panel: fraction of runs that reach the escape radius, pooled over systems
    a = ax[-1]
    multiples = [float(m) for m in RATE_MULTIPLES]
    escaped = []
    for multiple in RATE_MULTIPLES:
        final = np.concatenate(
            [np.array(d["distance"])[:, d["lanes"].index(rules.restored_lane_name(multiple)), -1] for d in records]
        )
        escaped.append(np.mean(final >= ESCAPED_ABOVE))
    a.axvline(1.0, color=MUTED, lw=0.9, ls=DASHED)
    n = sum(np.array(d["distance"]).shape[0] for d in records)
    z = 1.96
    for multiple, p in zip(multiples, escaped):  # Wilson 95% interval of a binomial proportion
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        a.plot([multiple, multiple], [max(centre - half, 0), min(centre + half, 1)], color=COLOUR[rules.SNAP1], lw=1.0)
    a.plot(multiples, escaped, color=COLOUR[rules.SNAP1], marker="s", ls="none", mfc="white", mew=1.0)
    a.set_xscale("log")
    a.set_xticks(multiples, list(RATE_MULTIPLES))
    a.xaxis.set_minor_locator(FixedLocator([]))
    a.set_ylim(-0.05, 1.05)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1])
    a.set_xlabel(r"tracking rate  $\beta/\beta_c$")
    a.set_ylabel("fraction of runs that escape")
    a.grid(True, axis="y")

    letters = PanelLetters()
    letters(ax[:1], dx=-28)
    letters(ax[1:-1], dx=-4)
    letters(ax[-1:], dx=-28)
    snap1 = COLOUR[rules.SNAP1]
    top_legend(
        fig,
        [
            Line2D([], [], color=COLOUR[rules.RTRL], label="RTRL"),
            Line2D([], [], color=COLOUR[rules.IMMEDIATE], label="immediate"),
            Line2D([], [], color=snap1, label="SnAp-1"),
            Line2D([], [], color=snap1, ls=":", label=r"SnAp-1 + restoration, $0.3\,\beta_c$"),
            Line2D([], [], color=snap1, ls="--", label=r"SnAp-1 + restoration, $10\,\beta_c$"),
        ],
    )
    save(fig, NAME)


if __name__ == "__main__":
    draw()
