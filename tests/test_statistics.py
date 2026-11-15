"""The statistics used in the tables."""

import numpy as np

from trace_instruments.statistics import bootstrap_median_interval, holm_adjust, wilcoxon_log_ratio


def test_holm_adjustment():
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    adjusted = holm_adjust([0.5, float("nan"), 0.9])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [1.0, 1.0])


def test_wilcoxon_log_ratio():
    assert np.isnan(wilcoxon_log_ratio([1.2, 1.3, 0.9]))  # fewer than six pairs
    assert np.isnan(wilcoxon_log_ratio(np.ones(10)))  # no difference at all
    ratios = np.array([1.5, 2.0, 1.2, 1.8, 1.1, 1.3, 1.6, 1.4])
    assert wilcoxon_log_ratio(ratios) < 0.05
    np.testing.assert_allclose(wilcoxon_log_ratio(ratios), wilcoxon_log_ratio(1 / ratios))


def test_bootstrap_interval_is_deterministic_and_brackets_the_median():
    x = np.random.default_rng(1).lognormal(size=40)
    low, high = bootstrap_median_interval(x)
    assert (low, high) == bootstrap_median_interval(x)
    assert low <= np.median(x) <= high
    assert all(np.isnan(bootstrap_median_interval([1.0, float("nan")])))
