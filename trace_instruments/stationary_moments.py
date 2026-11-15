"""Stationary moments of an update rule at a fixed parameter, and the quantities the theory reads off them."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from scipy.linalg import solve_continuous_lyapunov

from . import rules
from .config import DTYPE
from .teacher_student import TeacherStudentSystem
from .traces import TraceBank


class StationaryMoments(NamedTuple):
    """Time averages at a fixed student parameter; every field is a dict keyed by rule name except the last.

    cross_gram        ``C_Q = E[Z^T F J]``                                 (B, P, P)
    instrument_gram   ``Sigma_Q = E[Z^T F Z]``                             (B, P, P)
    mean_update       ``V_Q = E[Z^T delta]``                               (B, P)
    gradient_cosine   mean instantaneous cosine between ``Z^T delta`` and ``J^T delta``   (B,)
    mean_loss         mean of ``0.5 |err|^2``                              (B,)
    """

    cross_gram: dict
    instrument_gram: dict
    mean_update: dict
    gradient_cosine: dict
    mean_loss: torch.Tensor


@torch.no_grad()
def estimate_stationary_moments(system: TeacherStudentSystem, rule_names, num_steps, burn_in=500, theta=None):
    """Average the moments of every rule over ``num_steps`` steps after ``burn_in``.

    The exact rule is always included, so ``cross_gram["rtrl"]`` is the Gauss-Newton matrix ``G``.
    """
    rule_names = list(dict.fromkeys([rules.RTRL] + list(rule_names)))
    bank = TraceBank(system.student, rule_names)
    system.reset_state()
    P, B = system.student.P, system.B

    def zeros(*shape):
        return torch.zeros(*shape, dtype=DTYPE, device=system.device)

    C = {name: zeros(B, P, P) for name in rule_names}
    Sigma = {name: zeros(B, P, P) for name in rule_names}
    V = {name: zeros(B, P) for name in rule_names}
    cosine = {name: zeros(B) for name in rule_names}
    loss = zeros(B)
    F = system.readout_gram
    for t in range(num_steps + burn_in):
        delta, D, phi, W, err = system.step(theta)
        bank.step(D, phi, W)
        if t < burn_in:
            continue
        J = bank.Z[rules.RTRL]
        FJ = torch.bmm(F, J)
        exact_update = torch.einsum("bkp,bk->bp", J, delta)
        loss += 0.5 * (err * err).sum(1)
        for name in rule_names:
            Z = bank.Z[name]
            Zt = Z.transpose(1, 2)
            C[name] += torch.bmm(Zt, FJ)
            Sigma[name] += torch.bmm(Zt, torch.bmm(F, Z))
            update = torch.einsum("bkp,bk->bp", Z, delta)
            V[name] += update
            cosine[name] += (update * exact_update).sum(1) / (update.norm(dim=1) * exact_update.norm(dim=1) + 1e-300)
    f = 1.0 / num_steps
    return StationaryMoments(
        {name: C[name] * f for name in rule_names},
        {name: Sigma[name] * f for name in rule_names},
        {name: V[name] * f for name in rule_names},
        {name: cosine[name] * f for name in rule_names},
        loss * f,
    )


def coherence_spectrum(C, Sigma, G, rtol=1e-9):
    """Squared canonical correlations between instrument and sensitivity (NumPy, one system), ascending.

    Directions outside the range of ``Sigma`` have coherence zero.
    """
    gw, gv = np.linalg.eigh((G + G.T) / 2)
    keep = gw > rtol * gw.max()
    G_inv_half = gv[:, keep] / np.sqrt(gw[keep])
    sw, sv = np.linalg.eigh((Sigma + Sigma.T) / 2)
    keep_sigma = sw > rtol * max(sw.max(), 1e-300)
    Sigma_inv_half = sv[:, keep_sigma] / np.sqrt(sw[keep_sigma])
    whitened = Sigma_inv_half.T @ C @ G_inv_half  # whitened cross-Gram
    rho = np.linalg.svd(whitened, compute_uv=False)
    rho2 = np.zeros(keep.sum())
    rho2[: rho.size] = np.clip(rho**2, 0, 1)
    return np.sort(rho2)


def spectral_summary(C, Sigma, G):
    """Per-lane summaries of one rule: stability, rank, coherence spectrum and efficiency loss.

    Returns one dict per lane with keys ``min_real_eig`` and ``max_abs_eig`` (spectrum of ``C``), ``rank``,
    ``sym_min_eig`` (smallest eigenvalue of the symmetric part), ``coherence`` (squared canonical
    correlations) and ``efficiency_loss`` (``tr(G C^-1 Sigma C^-T) / P``, infinite when ``C`` is singular).
    """
    out = []
    C, Sigma, G = C.cpu().numpy(), Sigma.cpu().numpy(), G.cpu().numpy()
    P = C.shape[-1]
    for b in range(C.shape[0]):
        c, s, g = C[b], Sigma[b], G[b]
        eigenvalues = np.linalg.eigvals(c)
        singular_values = np.linalg.svd(c, compute_uv=False)
        rank = int((singular_values > 1e-8 * singular_values[0]).sum())
        summary = dict(
            min_real_eig=float(eigenvalues.real.min()),
            max_abs_eig=float(np.abs(eigenvalues).max()),
            rank=rank,
            sym_min_eig=float(np.linalg.eigvalsh((c + c.T) / 2).min()),
            coherence=coherence_spectrum(c, s, g),
            efficiency_loss=float("inf"),
        )
        if rank == P:
            c_inv = np.linalg.inv(c)
            summary["efficiency_loss"] = float(np.trace(g @ c_inv @ s @ c_inv.T) / P)
        out.append(summary)
    return out


def lyapunov_covariance(C, Sigma, step_size, noise_var, preconditioner=None):
    """Stationary parameter covariance of the constant-step rule (first order in the step size)."""
    if preconditioner is None:
        return solve_continuous_lyapunov(C, step_size * noise_var * Sigma)
    return solve_continuous_lyapunov(
        preconditioner @ C, step_size * noise_var * preconditioner @ Sigma @ preconditioner.T
    )
