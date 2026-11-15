"""Statistics for paired comparisons.

The unit of replication is the seed (initialisation and, for synthetic streams, the data) for the
prediction tasks and the independent system for teacher-student experiments.  Comparisons with RTRL are
paired: for each unit the ratio of the two errors.  Reported are the median ratio, a 95% bootstrap
interval for the median (10 000 resamples), and a two-sided Wilcoxon signed-rank test of
``log ratio = 0`` with Holm correction over all the comparisons of one table.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_median_interval(x, num_resamples=10_000):
    """95% percentile-bootstrap interval for the median of the finite entries of ``x``.

    The random generator is created per call (seed 0), so one sample does not affect the interval of
    another.  Fewer than three finite values give ``(nan, nan)``.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    medians = np.median(x[rng.integers(0, x.size, (num_resamples, x.size))], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def wilcoxon_log_ratio(ratios):
    """Two-sided Wilcoxon signed-rank p-value for ``log ratio = 0`` (NaN with fewer than six pairs)."""
    x = np.log(np.asarray(ratios, float))
    x = x[np.isfinite(x)]
    if x.size < 6 or np.allclose(x, 0):
        return float("nan")
    return float(wilcoxon(x).pvalue)


def holm_adjust(p_values):
    """Holm step-down adjustment of a family of p-values; NaN entries are left out of the family."""
    p = np.asarray(p_values, float)
    out = np.full(p.shape, np.nan)
    finite = np.where(np.isfinite(p))[0]
    order = finite[np.argsort(p[finite])]
    m = len(order)
    running_max = 0.0
    for j, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - j) * p[i]))
        out[i] = running_max
    return out
