"""Calibration restoration: tracking the bias field of an approximate rule.

A running estimate ``b`` of the mean-field bias ``D(theta) = E[(J - Z)^T delta]`` is added to the cheap
update ``Z^T delta``.  The estimators differ only in how they observe ``D``:

``projected`` / ``ema``
    ``k`` exact forward tangents ``tau_t = J_t U`` along directions ``U`` (orthonormal, ``P x k``) held
    fixed for a window; ``J`` is never formed.  The window average
    ``s = mean_t (tau_t - Z_t U)^T delta_t`` estimates ``U^T D`` after a burn-in that lets the restarted
    tangents forget their zero start.

    * ``projected``:  ``b <- b + beta U (s - U^T b)``         (sets the component of ``b`` inside span ``U``)
    * ``ema``:        ``b <- (1 - beta) b + beta (P / k) U s``  (unbiased over ``U``; variance amplified by ``P / k``)

    ``support`` (boolean mask over parameters) confines the directions to a known subspace.
``exact``
    full exact sensitivity (costs RTRL): ``b <- b + beta ((J - Z)^T delta - b)``.  Reference only.
``uoro``
    rank-one unbiased sensitivity estimate ``J~ = h~ w~^T`` (Tallec & Ollivier):
    ``b <- b + beta (w~ (h~ . delta) - Z^T delta - b)``.
"""

from __future__ import annotations

import torch

from .config import DTYPE

PROJECTED = "projected"
EMA = "ema"
EXACT = "exact"
UORO = "uoro"


class BiasFieldEstimator:
    """Running estimate ``self.estimate`` (B, P) of the bias field, advanced once per step."""

    def __init__(
        self, cell, B, device, mode=PROJECTED, num_tangents=4, window=200, burn_in=20, rate=0.1, seed=0, support=None
    ):
        self.cell, self.num_tangents, self.B, self.device, self.mode = cell, num_tangents, B, device, mode
        self.window, self.burn_in, self.rate = window, burn_in, rate
        self.generator = torch.Generator(device=device).manual_seed(seed + 999)
        self.estimate = torch.zeros(B, cell.P, dtype=DTYPE, device=device)
        self.support = None if support is None else support.to(device=device, dtype=DTYPE)
        # dimension of the subspace being probed
        self.probed_dim = cell.P if support is None else int(support.sum())
        if mode in (PROJECTED, EMA):
            self._start_window()
        elif mode == EXACT:
            self.J = torch.zeros(B, cell.n, cell.P, dtype=DTYPE, device=device)
        elif mode == UORO:
            self.state_factor = torch.zeros(B, cell.n, dtype=DTYPE, device=device)
            self.parameter_factor = torch.zeros(B, cell.P, dtype=DTYPE, device=device)
        else:
            raise ValueError(mode)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.generator, dtype=DTYPE, device=self.device)

    def _start_window(self):
        """Draw fresh orthonormal directions and restart the tangents and the window statistic."""
        g = self._randn(self.B, self.cell.P, self.num_tangents)
        if self.support is not None:
            g = g * self.support[None, :, None]
        self.directions = torch.linalg.qr(g).Q
        self.tangents = torch.zeros(self.B, self.cell.n, self.num_tangents, dtype=DTYPE, device=self.device)
        self.window_sum = torch.zeros(self.B, self.num_tangents, dtype=DTYPE, device=self.device)
        self.window_count = 0

    def reset_lanes(self, lanes):
        """Zero the state of the selected lanes (boolean mask)."""
        self.estimate[lanes] = 0
        for name in ("tangents", "window_sum", "J", "state_factor", "parameter_factor"):
            if hasattr(self, name):
                getattr(self, name)[lanes] = 0

    def __call__(self, t, Z, delta, D, phi, W):
        """Advance with the Jacobians of step ``t`` and return the current estimate."""
        cell = self.cell
        if self.mode in (PROJECTED, EMA):
            if t > 0 and t % self.window == 0:
                if self.window_count:
                    s = self.window_sum / self.window_count
                    U = self.directions
                    if self.mode == EMA:
                        scale = self.rate * (self.probed_dim / self.num_tangents)
                        self.estimate = (1 - self.rate) * self.estimate + scale * torch.bmm(U, s[:, :, None])[:, :, 0]
                    else:
                        residual = s - torch.bmm(U.transpose(1, 2), self.estimate[:, :, None])[:, :, 0]
                        self.estimate = self.estimate + self.rate * torch.bmm(U, residual[:, :, None])[:, :, 0]
                self._start_window()
            self.tangents = D[:, :, None] * torch.bmm(W, self.tangents) + cell.immediate_jacobian_times(
                D, phi, self.directions
            )
            if t % self.window >= self.burn_in:
                self.window_sum += torch.einsum("bnk,bn->bk", self.tangents - torch.bmm(Z, self.directions), delta)
                self.window_count += 1
        elif self.mode == EXACT:
            self.J = torch.bmm(D[:, :, None] * W, self.J) + cell.immediate_jacobian(D, phi)
            observed = torch.einsum("bnp,bn->bp", self.J - Z, delta)
            self.estimate = self.estimate + self.rate * (observed - self.estimate)
        else:  # uoro
            eps = 1e-12
            nu = torch.sign(self._randn(self.B, cell.n))
            propagated = D * torch.einsum("bij,bj->bi", W, self.state_factor)
            # nu^T B_t as a parameter vector: row i of the parameter matrix is nu_i D_i phi
            nu_B = cell.immediate_jacobian_transpose_times(D, phi, nu)
            r0 = torch.sqrt((self.parameter_factor.norm(dim=1) + eps) / (propagated.norm(dim=1) + eps))
            r1 = torch.sqrt((nu_B.norm(dim=1) + eps) / (nu.norm(dim=1) + eps))
            self.state_factor = r0[:, None] * propagated + r1[:, None] * nu
            self.parameter_factor = self.parameter_factor / r0[:, None] + nu_B / r1[:, None]
            observed = self.parameter_factor * (self.state_factor * delta).sum(1, keepdim=True) - torch.einsum(
                "bnp,bn->bp", Z, delta
            )
            self.estimate = self.estimate + self.rate * (observed - self.estimate)
        return self.estimate


class LaneGroupedEstimators:
    """Several estimators, each acting on its own subset of lanes.

    ``groups`` is a list of ``(estimator, lane_index)`` pairs; lanes that belong to no group receive a
    zero correction.
    """

    def __init__(self, groups, B, P, device):
        self.groups = groups
        self.estimate = torch.zeros(B, P, dtype=DTYPE, device=device)

    def __call__(self, t, Z, delta, D, phi, W):
        for estimator, lanes in self.groups:
            self.estimate[lanes] = estimator(t, Z[lanes], delta[lanes], D[lanes], phi[lanes], W[lanes])
        return self.estimate

    def reset_lanes(self, lanes):
        for estimator, index in self.groups:
            estimator.reset_lanes(lanes[index])
