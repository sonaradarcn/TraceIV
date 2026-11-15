"""Figure ``tasks`` (Section 9.8).

Paired ratios to RTRL on the prediction tasks: median and bootstrap 95% interval; markers are filled when
the difference is significant after Holm correction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from analysis import prediction_tasks as analysis
from analysis.common import load
from plotting.style import COLOUR, GRID, LEAK_05, MARKER, MUTED, TBPTT_8, TEXT_WIDTH, PanelLetters, rule_handle
from plotting.style import save, top_legend
from trace_instruments import cells, rules
from trace_instruments.paths import results_dir
from trace_instruments.statistics import bootstrap_median_interval, holm_adjust, wilcoxon_log_ratio

NAME = "tasks"
RULES = [rules.SNAP1, rules.IMMEDIATE, LEAK_05, rules.UORO, TBPTT_8]
TASK_LABELS = ["Delay-5", "NARMA-10", "Mackey–Glass", "Lorenz", "Sunspot"]
NUM_UNITS = 16
CELL_TITLES = {cells.TANH: "tanh cell", cells.GRU_RESET_AFTER: "gated cell"}


def draw():
    fig, ax = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.3), sharey=True)
    for c, cell in enumerate(CELL_TITLES):
        records = [
            load(results_dir(analysis.RESULTS_SUBDIR, f"{cell}_n{NUM_UNITS}_{tag}.json")) for tag in analysis.TASK_TAGS
        ]
        scratch, handover = {}, {}
        for ti, d in enumerate(records):
            selected = analysis.select_scratch_steps(d)
            mse = np.array(d["scratch"]["mse"], float)
            for rule in RULES:
                r = d["rules"].index(rule)
                scratch[(rule, ti)] = mse[:, selected[r], r] / mse[:, selected[0], 0]
                handover[(rule, ti)] = analysis.handover_ratios(d, r)[0]
        keys = [(rule, ti) for ti in range(len(records)) for rule in RULES]
        adjusted = holm_adjust(
            [wilcoxon_log_ratio(scratch[k]) for k in keys] + [wilcoxon_log_ratio(handover[k]) for k in keys]
        )
        for b, ratios in enumerate((scratch, handover)):
            a = ax[b, c]
            a.axvline(1, color=MUTED, lw=0.9)
            x_low, x_high = (0.3, 600) if b == 0 else (0.5, 8)
            for i, (rule, ti) in enumerate(keys):
                x = ratios[(rule, ti)]
                if np.isnan(x).sum() > len(x) // 2:
                    continue
                y = ti + (RULES.index(rule) - 2) * 0.16
                low, high = bootstrap_median_interval(x)
                median = np.nanmedian(x)
                significant = adjusted[b * len(keys) + i] < 0.05
                colour = COLOUR[rule]
                if median > x_high:  # off the axis: arrow at the edge, value written beside it
                    a.annotate(
                        "",
                        xy=(x_high, y),
                        xytext=(x_high / 1.35, y),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.0, mutation_scale=6),
                    )
                    a.annotate(f"{median:.0f}", (x_high / 1.4, y), ha="right", va="center", color=colour)
                    continue
                a.plot([max(low, x_low), min(high, x_high)], [y, y], color=colour, lw=1.0, solid_capstyle="butt")
                edge = dict(color=colour, ls="none", markersize=3.5, mew=0, clip_on=False, zorder=4)
                if high > x_high:  # interval runs off the axis
                    a.plot([x_high], [y], marker=">", **edge)
                if low < x_low:
                    a.plot([x_low], [y], marker="<", **edge)
                a.plot(
                    [median],
                    [y],
                    color=colour,
                    marker=MARKER[rule],
                    ls="none",
                    mfc=colour if significant else "white",
                    mew=0.8,
                    markersize=4,
                )
            a.set_xscale("log")
            a.set_xlim(x_low, x_high)
            ticks = [1, 10, 100] if b == 0 else [0.5, 1, 2, 5]
            a.set_xticks(ticks, [f"{v:g}" for v in ticks])
            a.xaxis.set_minor_locator(FixedLocator([]))
            a.grid(True, axis="x")
            a.set_title(
                CELL_TITLES[cell] + (", from a random start" if b == 0 else ", after handover"),
                loc="right",
                color=MUTED,
            )
            a.set_xlabel("held-out MSE / RTRL" if b == 0 else "held-out MSE / RTRL continued")
            for ti in range(1, len(records)):
                a.axhline(ti - 0.5, color=GRID, lw=0.5)
            a.tick_params(axis="y", length=0)
    ax[0, 0].set_yticks(range(len(TASK_LABELS)), TASK_LABELS)
    ax[0, 0].set_ylim(len(TASK_LABELS) - 0.5, -0.5)
    letters = PanelLetters()
    letters(ax[0, :1], dx=-52)
    letters(ax[0, 1:], dx=-6)
    letters(ax[1, :1], dx=-52)
    letters(ax[1, 1:], dx=-6)
    top_legend(
        fig,
        [rule_handle(rule, line=False) for rule in RULES]
        + [
            Line2D([], [], color=MUTED, marker="o", ls="none", mfc=MUTED, label=r"Holm-adjusted $p<0.05$"),
            Line2D([], [], color=MUTED, marker="o", ls="none", mfc="white", label=r"$p\geq0.05$"),
        ],
    )
    save(fig, NAME)


if __name__ == "__main__":
    draw()
