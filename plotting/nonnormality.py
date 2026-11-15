"""Figure ``nonnormal`` (Section 9.1).

At fixed spectrum, non-normality removes ``sym(C) > 0`` and inflates the efficiency loss (a-c); spectrum of
the cross-Gram in the exact SnAp-1 example (d).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import load
from plotting.style import COLOUR, DASHED, INK, MARKER, MUTED, TEXT_WIDTH, PanelLetters, median_band, rule_handle
from plotting.style import save, top_legend
from matplotlib.ticker import FixedLocator
from trace_instruments import rules
from trace_instruments.paths import results_dir

NAME = "nonnormal"


def draw():
    sweep = load(results_dir("linear_dynamics", "closed_forms.json"))["nonnormality_sweep"]
    scales = sorted({row["upper_scale"] for row in sweep})

    def column(scale, value):
        return [value(row) for row in sweep if row["upper_scale"] == scale]

    radius = np.array([np.median(column(s, lambda row: row["numerical_radius"])) for s in scales])
    fig, ax = plt.subplots(1, 4, figsize=(TEXT_WIDTH, 2.15))
    for rule in (rules.IMMEDIATE, rules.SNAP1):
        for axis, key in zip(ax, ("sym_min_eig", "min_real_eig_whitened", "efficiency_loss")):
            median_band(axis, radius, [column(s, lambda row: row["rules"][rule][key]) for s in scales], rule)
    for a in ax[:3]:
        a.axvline(1.0, color=MUTED, lw=0.7, ls=DASHED)
        a.set_xlabel(r"numerical radius $w(A)$")
        a.grid(True, axis="y")
        a.set_xlim(0.6, 3.4)
        a.set_xticks([1, 2, 3])
    for a in ax[:2]:
        a.axhline(0, color=MUTED, lw=0.6)
    ax[0].set_yscale("symlog", linthresh=1.0)
    ax[0].set_yticks([-1e4, -1e2, -1, 0, 1], [r"$-10^4$", r"$-10^2$", r"$-1$", "0", "1"])
    ax[0].set_ylim(-2e5, 3)
    ax[0].yaxis.set_minor_locator(FixedLocator([]))
    ax[1].set_ylim(-0.05, 0.8)
    ax[2].set_ylim(0.9, 200)
    ax[2].set_yticks([1, 10, 100], ["1", "10", "100"])
    ax[2].yaxis.set_minor_locator(FixedLocator([]))
    ax[0].set_ylabel(r"$\lambda_{\min}\,\mathrm{sym}(C)$")
    ax[1].set_ylabel(r"$\min\,\mathrm{Re}\,\mathrm{spec}(\Sigma_Z^{-1}C)$")
    ax[2].set_yscale("log")
    ax[2].set_ylabel(r"efficiency loss  $P^{-1}\sum_i\rho_i^{-2}$")

    # d: real parts of the eigenvalues, sorted; open markers are complex pairs
    example = load(results_dir("linear_dynamics", "snap1_exact_instability.json"))
    a = ax[3]
    a.axhspan(-1e5, 0, color=COLOUR[rules.SNAP1], alpha=0.07, lw=0)
    a.axhline(0, color=MUTED, lw=0.8)
    for rule, key, dx in ((rules.IMMEDIATE, "eigenvalues_immediate", -0.12), (rules.SNAP1, "eigenvalues_snap1", 0.12)):
        z = np.array(example[key])
        z = z[np.argsort(z[:, 0])]
        i = np.arange(1, len(z) + 1) + dx
        is_complex = np.abs(z[:, 1]) > 1e-9
        a.scatter(
            i[~is_complex], z[~is_complex, 0], s=14, color=COLOUR[rule], marker=MARKER[rule], linewidths=0, zorder=3
        )
        a.scatter(
            i[is_complex],
            z[is_complex, 0],
            s=14,
            facecolors="white",
            edgecolors=COLOUR[rule],
            marker=MARKER[rule],
            linewidths=0.9,
            zorder=3,
        )
        if rule == rules.SNAP1:
            a.scatter([i[0]], [z[0, 0]], s=70, facecolors="none", edgecolors=INK, linewidths=0.8, zorder=4)
            a.annotate(
                r"$\mathrm{Re}\,\lambda<0$", (i[0], z[0, 0]), textcoords="offset points", xytext=(8, -1), va="center"
            )
    a.set_yscale("symlog", linthresh=1.0)
    a.set_ylim(-2e4, 2e4)
    a.set_yticks([-1e3, -10, 0, 10, 1e3], [r"$-10^3$", r"$-10$", "0", r"$10$", r"$10^3$"])
    a.yaxis.set_minor_locator(FixedLocator([]))
    a.set_xticks([1, 3, 5, 7, 9])
    a.set_xlim(0.3, 9.7)
    a.grid(True, axis="y")
    a.set_xlabel(r"eigenvalue rank (by $\mathrm{Re}\,\lambda$)")
    a.set_ylabel(r"$\mathrm{Re}\,\lambda(C)$")
    ax[0].set_ylabel(r"$\lambda_{\min}\,\mathrm{sym}(C)$")
    PanelLetters()(ax)
    top_legend(fig, [rule_handle(rules.IMMEDIATE), rule_handle(rules.SNAP1)])
    save(fig, NAME)


if __name__ == "__main__":
    draw()
