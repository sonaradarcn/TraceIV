"""Figure ``frontier`` (Section 9.6).

Excess risk against arithmetic per step for restoration by tangent probes (one panel per network size, with
half of the units read out), and measured time per step of the recursions (last panel).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.common import load, load_all
from plotting.style import COLOUR, HYBRID_COLOUR, MARKER, MUTED, PROBE_STYLE, TBPTT_8, TEXT_WIDTH, PanelLetters
from plotting.style import no_minor_ticks, rule_handle, save, top_legend
from trace_instruments import rules
from trace_instruments.paths import results_dir

NAME = "frontier"
TIMED_RULES = (rules.RTRL, rules.SNAP1, rules.IMMEDIATE, rules.UORO, TBPTT_8, "probes:16")


def draw():
    records = [d for d in load_all("restoration", "cost_frontier") if d["num_readout"] * 2 == d["num_units"]]
    records.sort(key=lambda d: d["num_units"])
    fig, ax = plt.subplots(1, len(records) + 1, figsize=(TEXT_WIDTH, 2.3))
    for i, (a, d) in enumerate(zip(ax, records)):
        excess = np.array(d["excess_risk"], float)
        median = np.nanmedian(excess, 0)
        cost = np.array(d["arithmetic_cost"], float) / d["arithmetic_cost"][0]
        lanes = d["lanes"]
        num_blind = d["num_blind_parameters"]

        def usable_median(j):
            return median[j] if np.isfinite(median[j]) and not np.isnan(excess[:, j]).any() else np.inf

        # best setting of tracking rate and window for every number of tangents
        tangents = sorted({lane["probes"]["num_tangents"] for lane in lanes[3:]})
        xs, ys = [], []
        for k in tangents:
            group = [j for j in range(3, len(lanes)) if lanes[j]["probes"]["num_tangents"] == k]
            best = min(group, key=usable_median)
            xs.append(cost[best])
            ys.append(median[best])
        a.plot(xs, ys, **PROBE_STYLE)
        for j, (x, y, k) in enumerate(zip(xs, ys, tangents)):
            a.annotate(
                "$P_e$" if k == num_blind else f"{k}",
                (x, y),
                textcoords="offset points",
                xytext=(-3, 5) if j == 0 else (9, 5) if k == num_blind else (5, 2),
                ha="right" if j == 0 else "left",
                color=MUTED,
            )
        a.scatter([cost[1]], [median[1]], color=COLOUR[rules.SNAP1], marker="s", s=20, zorder=3)
        a.scatter([cost[2]], [median[2]], color=HYBRID_COLOUR, marker="*", s=42, zorder=3)
        a.scatter([cost[0]], [median[0]], color=COLOUR[rules.RTRL], marker="o", s=20, zorder=3)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlim(1e-3, 8)
        a.set_xticks([1e-3, 1e-2, 1e-1, 1])
        a.set_ylim(5e-6, 3e-2)
        a.set_yticks([1e-5, 1e-4, 1e-3, 1e-2])
        a.grid(True)
        a.set_xlabel("arithmetic per step / RTRL")
        a.set_title(f"$n={d['num_units']}$,  $P_e={num_blind}$", loc="right", color=MUTED)
        if i:
            a.set_yticklabels([])
    ax[0].set_ylabel("excess risk")

    benchmark = load(results_dir("restoration", "step_cost", "cost_cpu.json"))
    a = ax[-1]
    for rule in TIMED_RULES:
        rows = sorted([row for row in benchmark["rows"] if row["rule"] == rule], key=lambda row: row["num_units"])
        style = PROBE_STYLE if rule.startswith("probes") else dict(color=COLOUR[rule], marker=MARKER[rule])
        a.plot([row["num_units"] for row in rows], [row["microseconds_per_step"] / 1e3 for row in rows], **style)
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.grid(True)
    a.set_xlabel("units  $n$")
    a.set_ylabel("recursion time (ms per step)")
    a.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    a.set_ylim(0.08, 500)
    a.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    no_minor_ticks(ax)
    PanelLetters()([ax[0], ax[-1]], dx=-28)
    top_legend(
        fig,
        [
            rule_handle(rules.RTRL, line=False),
            rule_handle(rules.SNAP1, line=False),
            Line2D([], [], label="SnAp-1 + $k$ probes", **PROBE_STYLE),
            Line2D([], [], color=HYBRID_COLOUR, marker="*", ls="none", markersize=6, label="exact on unread units"),
            rule_handle(rules.IMMEDIATE),
            rule_handle(rules.UORO),
            rule_handle(TBPTT_8),
        ],
    )
    save(fig, NAME)


if __name__ == "__main__":
    draw()
