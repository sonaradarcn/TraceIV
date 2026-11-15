"""Exact moments of the linear model: identities that hold in closed form."""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from trace_instruments import rules
from trace_instruments.linear_theory import exact_moments, numerical_radius, propagation_matrix
from trace_instruments.linear_theory import restored_map_radius, schur_form_matrix, summarise_rule

A = schur_form_matrix(seed=33, upper_scale=1.0)


def test_schur_form_fixes_the_spectrum():
    spectrum = np.sort(np.linalg.eigvals(A).real)
    for upper_scale in (0.0, 2.0, 4.0):
        other = np.sort(np.linalg.eigvals(schur_form_matrix(33, upper_scale)).real)
        np.testing.assert_allclose(other, spectrum, atol=1e-10)
    assert np.abs(spectrum).max() < 1


def test_rtrl_is_the_efficient_instrument():
    summary, C, Sigma, Pi = summarise_rule(A, rules.RTRL)
    np.testing.assert_allclose(C, C.T, atol=1e-9)  # the Gauss-Newton matrix
    np.testing.assert_allclose(C, Sigma, atol=1e-9)
    assert np.linalg.eigvalsh((C + C.T) / 2).min() > 0
    np.testing.assert_allclose(summary["efficiency_loss"], 1.0, rtol=1e-8)
    np.testing.assert_allclose(summary["coherence_min"], 1.0, rtol=1e-8)
    np.testing.assert_allclose(Pi, solve_discrete_lyapunov(A, np.eye(4)))


def test_immediate_rule_instrument_gram():
    """Without trace memory the instrument is the regressor itself: Sigma = I (x) Pi."""
    _, Sigma, Pi = exact_moments(A, propagation_matrix(A, rules.IMMEDIATE))
    np.testing.assert_allclose(Sigma, np.kron(np.eye(4), Pi), atol=1e-12)


def test_efficiency_loss_is_at_least_one():
    for rule in (rules.SNAP1, rules.IMMEDIATE, "leak:0.5"):
        summary = summarise_rule(A, rule)[0]
        assert summary["efficiency_loss"] >= 1 - 1e-9
        assert 0 <= summary["coherence_min"] <= summary["coherence_median"] <= 1 + 1e-9


def test_unstable_trace_is_reported():
    summary, C, _, _ = summarise_rule(A, "leak:1.0")
    assert summary["trace_unstable"] and C is None and np.isnan(summary["efficiency_loss"])


def test_numerical_radius():
    np.testing.assert_allclose(numerical_radius(np.array([[0.0, 1.0], [0.0, 0.0]])), 0.5, rtol=1e-6)
    symmetric = (A + A.T) / 2
    np.testing.assert_allclose(numerical_radius(symmetric), np.abs(np.linalg.eigvalsh(symmetric)).max(), rtol=1e-6)
    assert numerical_radius(A) >= np.abs(np.linalg.eigvals(A)).max() - 1e-9


def test_restored_map_of_an_unbiased_rule():
    """With C = G the map is block triangular: spectrum of I - alpha G together with 1 - beta."""
    G = exact_moments(A, A)[0]
    step_size, tracking_rate = 0.5 / np.linalg.eigvalsh(G).max(), 0.3
    expected = max(np.abs(1 - step_size * np.linalg.eigvalsh(G)).max(), 1 - tracking_rate)
    np.testing.assert_allclose(restored_map_radius(G, G, step_size, tracking_rate), expected, rtol=1e-9)
