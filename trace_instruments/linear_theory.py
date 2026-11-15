"""Exact moments for linear recurrent dynamics (no simulation).

Model: ``h_t = A h_{t-1} + x_t`` with white input, full read-out and ``theta = vec(A)``.  A trace with
propagation matrix ``L`` (``L = A`` for RTRL, ``diag(A)`` for SnAp-1, ``0`` for the immediate rule,
``rate * I`` for a leak) satisfies ``Z_t E = sum_k L^k E h_{t-1-k}``, so

    C     = sum_{k,l} (L^k^T A^l) (x) Gamma(k - l)^T
    Sigma = sum_{k,l} (L^k^T L^l) (x) Gamma(k - l)^T,        Gamma(s) = E[h_{t+s} h_t^T].
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import rules


def lag_covariance(Pi, lag, powers_of_A):
    """``Gamma(lag) = E[h_{t+lag} h_t^T]`` from the stationary covariance ``Pi`` and the powers of ``A``."""
    return powers_of_A[lag] @ Pi if lag >= 0 else Pi @ powers_of_A[-lag].T


def exact_moments(A, L, tol=1e-13, max_terms=4000):
    """Cross-Gram ``C``, instrument Gram ``Sigma`` and state covariance ``Pi`` for propagation matrix ``L``.

    The series are truncated once the powers of ``A`` fall below ``tol`` in magnitude.
    """
    n = A.shape[0]
    Pi = solve_discrete_lyapunov(A, np.eye(n))
    powers_A, powers_L = [np.eye(n)], [np.eye(n)]
    while np.abs(powers_A[-1]).max() > tol and len(powers_A) < max_terms:
        powers_A.append(powers_A[-1] @ A)
    while np.abs(powers_L[-1]).max() > tol and len(powers_L) < len(powers_A):
        powers_L.append(powers_L[-1] @ L)
    K = len(powers_A)
    powers_L += [np.zeros((n, n))] * (K - len(powers_L))
    C = np.zeros((n * n, n * n))
    Sigma = np.zeros_like(C)
    gamma = {s: lag_covariance(Pi, s, powers_A).T for s in range(-K + 1, K)}
    for k in range(K):
        if not powers_L[k].any():
            break
        for l in range(K):
            g = gamma[k - l]
            C += np.kron(powers_L[k].T @ powers_A[l], g)
            if powers_L[l].any():
                Sigma += np.kron(powers_L[k].T @ powers_L[l], g)
    return C, Sigma, Pi


def numerical_radius(A, num_angles=720):
    """Numerical radius ``w(A) = max_theta lambda_max(sym(e^{i theta} A))`` on a grid of angles."""
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    return max(np.linalg.eigvalsh((np.exp(1j * t) * A + np.exp(-1j * t) * A.conj().T) / 2).max() for t in angles)


def propagation_matrix(A, rule):
    """Propagation matrix ``L`` of a rule for the linear model."""
    n = A.shape[0]
    L = {rules.RTRL: A, rules.SNAP1: np.diag(np.diag(A)), rules.IMMEDIATE: np.zeros((n, n))}.get(rule)
    if L is None:
        L = rules.leak_rate(rule) * np.eye(n)
    return L


def summarise_rule(A, rule):
    """Stability and efficiency of a rule on the linear system ``A``.

    Returns ``(summary, C, Sigma, Pi)``.  The summary holds the smallest eigenvalue of ``sym(C)``, the
    smallest real part of the spectrum of ``C`` and of ``Sigma^-1 C``, the efficiency loss
    ``tr(G C^-1 Sigma C^-T) / P`` and the smallest and median squared canonical correlation.  When the trace
    recursion itself is unstable the entries are NaN and the matrices are ``None``.
    """
    n = A.shape[0]
    L = propagation_matrix(A, rule)
    if np.abs(np.linalg.eigvals(L)).max() >= 0.999:  # the trace recursion itself is unstable
        nan = float("nan")
        summary = dict(
            sym_min_eig=nan,
            min_real_eig=nan,
            min_real_eig_whitened=nan,
            efficiency_loss=nan,
            coherence_min=nan,
            coherence_median=nan,
            trace_unstable=True,
        )
        return summary, None, None, None
    C, Sigma, Pi = exact_moments(A, L)
    G, _, _ = exact_moments(A, A)
    whitened_spectrum = np.linalg.eigvals(np.linalg.solve(Sigma, C))
    C_inv = np.linalg.inv(C)
    gw, gv = np.linalg.eigh((G + G.T) / 2)
    G_inv_half = gv / np.sqrt(gw) @ gv.T
    sw, sv = np.linalg.eigh((Sigma + Sigma.T) / 2)
    Sigma_inv_half = sv / np.sqrt(sw) @ sv.T
    rho = np.linalg.svd(Sigma_inv_half @ C @ G_inv_half, compute_uv=False)
    summary = dict(
        sym_min_eig=float(np.linalg.eigvalsh((C + C.T) / 2).min()),
        min_real_eig=float(np.linalg.eigvals(C).real.min()),
        min_real_eig_whitened=float(whitened_spectrum.real.min()),
        efficiency_loss=float(np.trace(G @ C_inv @ Sigma @ C_inv.T) / (n * n)),
        coherence_min=float(rho.min() ** 2),
        coherence_median=float(np.median(rho) ** 2),
    )
    return summary, C, Sigma, Pi


def schur_form_matrix(seed, upper_scale, n=4):
    """``A = Q (diag(lambda) + upper_scale * N) Q^T`` with ``N`` strictly upper triangular.

    The spectrum is fixed by ``seed``; ``upper_scale`` sets the departure from normality.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = rng.uniform(0.3, 0.8, n) * rng.choice([-1, 1], n)
    N = np.triu(rng.standard_normal((n, n)), 1) / np.sqrt(n)
    return Q @ (np.diag(lam) + upper_scale * N) @ Q.T


def restored_map_radius(C, G, step_size, tracking_rate):
    """Spectral radius of the linearised map of a rule with ideal restoration,

        [[I - step_size C, -step_size I], [tracking_rate (G - C), (1 - tracking_rate) I]],

    acting on (parameter error, estimate of the bias field).
    """
    P = C.shape[0]
    I = np.eye(P)
    M = np.block([[I - step_size * C, -step_size * I], [tracking_rate * (G - C), (1 - tracking_rate) * I]])
    return float(np.abs(np.linalg.eigvals(M)).max())
