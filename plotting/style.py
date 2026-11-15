"""Shared style of the figures.

Every figure is drawn at the text width of the paper (``TEXT_WIDTH`` inches) and included unscaled, so
that type has the same size in all of them: 7 pt for labels, ticks, legends and annotations, 8 pt bold
for panel letters.  Legends sit outside the axes, above the panels.  One colour and one marker per rule
throughout.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

from trace_instruments import rules  # noqa: E402
from trace_instruments.paths import figures_dir  # noqa: E402

TEXT_WIDTH = 494.5 / 72.27  # \textwidth of the paper, inches
FONT_SIZE = 7
INK, MUTED, GRID = "#111111", "#555555", "#e6e6e6"

LEAK_05, LEAK_09, TBPTT_8 = "leak:0.5", "leak:0.9", "tbptt:8"
COLOUR = {
    rules.RTRL: "#2878B5",
    rules.SNAP1: "#D55E00",
    rules.IMMEDIATE: "#009E73",
    LEAK_05: "#E69F00",
    LEAK_09: "#CC79A7",
    rules.UORO: "#6A51A3",
    TBPTT_8: "#555555",
}
HYBRID_COLOUR = "#111111"
MARKER = {
    rules.RTRL: "o",
    rules.SNAP1: "s",
    rules.IMMEDIATE: "^",
    LEAK_05: "D",
    LEAK_09: "v",
    rules.UORO: "P",
    TBPTT_8: "X",
}
LABEL = {
    rules.RTRL: "RTRL",
    rules.SNAP1: "SnAp-1",
    rules.IMMEDIATE: "immediate",
    LEAK_05: "leak 0.5",
    LEAK_09: "leak 0.9",
    rules.UORO: "UORO",
    TBPTT_8: "TBPTT-8",
}
# SnAp-1 with restoration by tangent probes: the colour of SnAp-1, open diamonds on a dashed line
PROBE_STYLE = dict(color=COLOUR[rules.SNAP1], marker="D", ls="--", mfc="white", mew=0.9, lw=1.0)
DASHED = (0, (3, 2))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial:italic",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,
        "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.h_pad": 0.03,
    }
)


class PanelLetters:
    """Writes consecutive bold letters (a, b, c, ...) above the top-left corner of panels."""

    ALPHABET = "abcdefgh"

    def __init__(self):
        self.count = 0

    def __call__(self, axes, dx=-24):
        for axis in axes:
            axis.annotate(
                self.ALPHABET[self.count],
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(dx, 5),
                textcoords="offset points",
                fontsize=FONT_SIZE + 1,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
            self.count += 1


def rule_handle(rule, line=True, label=None, **kwargs):
    """Legend entry of a rule."""
    return Line2D(
        [],
        [],
        color=COLOUR[rule],
        marker=MARKER[rule],
        ls="-" if line else "none",
        label=label or LABEL[rule],
        **kwargs,
    )


def top_legend(fig, handles, ncol=None):
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol or len(handles))


def median_band(axis, x, samples, rule, **kwargs):
    """Median line with interquartile band; ``samples`` has one row per ``x``."""
    samples = [np.asarray(y, float) for y in samples]
    enough = [np.isfinite(y).sum() >= 5 for y in samples]

    def statistic(function):
        return np.array([function(y) if ok else np.nan for y, ok in zip(samples, enough)])

    median = statistic(np.nanmedian)
    low = statistic(lambda y: np.nanpercentile(y, 25))
    high = statistic(lambda y: np.nanpercentile(y, 75))
    axis.fill_between(x, low, high, color=COLOUR[rule], alpha=0.18, lw=0)
    axis.plot(x, median, color=COLOUR[rule], marker=MARKER[rule], **kwargs)


def no_minor_ticks(axes):
    for axis in axes:
        axis.xaxis.set_minor_locator(FixedLocator([]))
        axis.yaxis.set_minor_locator(FixedLocator([]))


def save(fig, name):
    """Write ``<figures>/<name>.pdf`` and a PNG preview."""
    for axis in fig.axes:
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_minor_formatter(NullFormatter())
    out = figures_dir()
    (out / "preview").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / "preview" / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)
