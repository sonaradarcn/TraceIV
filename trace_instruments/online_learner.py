"""Online learning on prediction streams with a trained linear read-out.

The recurrent parameters ``theta`` (cell with bias) are updated by a lane-specific rule; the read-out
``(R, c0)`` is updated by its own exact gradient, which is immediate for every rule.  Rules per lane:

``rtrl`` | ``snap1`` | ``immediate`` | ``leak:<rate>`` | ``uoro`` | ``tbptt:<K>``

(``uoro`` keeps a rank-one unbiased sensitivity estimate and no trace; ``tbptt:<K>`` uses the sensitivity
truncated to the last ``K`` steps, i.e. the gradient of truncated backpropagation at every step),
optionally with restoration (:mod:`trace_instruments.restoration`) on lanes selected by a mask.
"""

from __future__ import annotations

import math

import torch

from . import rules
from .cells import GRU_RESET_AFTER, TANH, make_cell
from .config import DIVERGENCE_BOUND, DTYPE


class OnlineLearner:
    """Lane-batched online learner on a :class:`~trace_instruments.prediction_streams.PredictionStream`.

    Parameters
    ----------
    stream
        Source of ``(input, target)`` pairs; fixes the number of lanes ``B``.
    num_units
        Number of recurrent units.
    rule_names
        One rule name, or a list with one rule per lane.
    step_size, readout_step_size
        Step of the recurrent parameters (scalar or ``(B,)``) and of the read-out (defaults to the former).
    seed, gain
        Initialisation: recurrent weights ``gain * N(0, 1) / sqrt(n)``, input weights ``N(0, 1)``, zero
        biases and a zero read-out.
    replicas
        Lanes per seed; the lanes of a seed share data and initialisation.
    restoration_factory, restoration_mask
        Optional ``factory(cell, B, device)`` returning a bias-field estimator, and the lanes that use it.
    clip
        Bound on the norm of the recurrent update (``0`` disables clipping).
    cell
        ``tanh`` or ``gru_reset_after``.
    """

    def __init__(
        self,
        stream,
        num_units,
        rule_names,
        step_size,
        device,
        seed=0,
        gain=0.9,
        readout_step_size=None,
        restoration_factory=None,
        restoration_mask=None,
        clip=1.0,
        replicas=None,
        cell=TANH,
    ):
        n = num_units
        self.stream, self.device, self.B, self.n = stream, device, stream.B, n
        self.cell = make_cell(n, 1, self.B, device, cell, bias=True)
        generator = torch.Generator().manual_seed(seed)
        replicas = replicas or 1
        num_seeds = self.B // replicas  # each seed has `replicas` lanes sharing data and initialisation
        if cell == GRU_RESET_AFTER:
            # every gate: recurrent part gain / sqrt(n), input part N(0, 1), bias 0
            theta = torch.zeros(num_seeds, n, 3, n + 2, dtype=DTYPE)
            theta[..., :n] = gain * torch.randn(num_seeds, n, 3, n, generator=generator, dtype=DTYPE) / math.sqrt(n)
            theta[..., n] = torch.randn(num_seeds, n, 3, generator=generator, dtype=DTYPE)
            self.cell.theta = theta.reshape(num_seeds, -1).repeat_interleave(replicas, 0).to(device)
        else:
            W = (
                gain * torch.randn(num_seeds, n, n, generator=generator, dtype=DTYPE) / math.sqrt(n)
            ).repeat_interleave(replicas, 0)
            V = torch.randn(num_seeds, n, 1, generator=generator, dtype=DTYPE).repeat_interleave(replicas, 0)
            self.cell.set_weights(W.to(device), V.to(device))
        self.theta = self.cell.theta.clone()
        self.R = torch.zeros(self.B, 1, n, dtype=DTYPE, device=device)
        self.c0 = torch.zeros(self.B, 1, dtype=DTYPE, device=device)
        self.rule_names = rule_names if isinstance(rule_names, list) else [rule_names] * self.B
        self.set_trace_rules(self.rule_names)
        self.uoro_mask = self._lane_tensor(lambda name: 1.0 if name == rules.UORO else 0.0)
        self.tbptt_mask = self._lane_tensor(lambda name: 1.0 if rules.is_tbptt(name) else 0.0)
        self.tbptt_window = max([rules.tbptt_window(name) for name in self.rule_names if rules.is_tbptt(name)] + [0])
        self.jacobian_buffer = []
        self.step_size = torch.as_tensor(step_size, dtype=DTYPE, device=device).expand(self.B).clone()
        if readout_step_size is None:
            self.readout_step_size = self.step_size.clone()
        else:
            self.readout_step_size = (
                torch.as_tensor(readout_step_size, dtype=DTYPE, device=device).expand(self.B).clone()
            )
        self.restoration = restoration_factory(self.cell, self.B, device) if restoration_factory else None
        self.restoration_mask = restoration_mask
        self.clip = clip
        self.generator = torch.Generator(device=device).manual_seed(seed + 4321)
        self.reset_state()
        self.alive = torch.ones(self.B, dtype=torch.bool, device=device)

    def _lane_tensor(self, value_of, rule_names=None):
        names = self.rule_names if rule_names is None else rule_names
        return torch.tensor([value_of(name) for name in names], dtype=DTYPE, device=self.device)

    def set_trace_rules(self, rule_names):
        """Choose the trace recursion of every lane (coefficients of ``A``, ``diag(A)`` and the identity)."""
        self.exact_coef = self._lane_tensor(lambda name: 1.0 if name == rules.RTRL else 0.0, rule_names)
        self.snap_coef = self._lane_tensor(lambda name: 1.0 if name == rules.SNAP1 else 0.0, rule_names)
        self.leak_coef = self._lane_tensor(
            lambda name: rules.leak_rate(name) if rules.is_leak(name) else 0.0, rule_names
        )

    def reset_state(self):
        """Zero the hidden state, the trace, the UORO factors and the buffer of past Jacobians."""

        def zeros(*shape):
            return torch.zeros(*shape, dtype=DTYPE, device=self.device)

        self.h = zeros(self.B, self.n)
        self.Z = zeros(self.B, self.n, self.cell.P)
        self.uoro_state = zeros(self.B, self.n)
        self.uoro_parameters = zeros(self.B, self.cell.P)
        self.jacobian_buffer = []

    @torch.no_grad()
    def forward(self, x, y, theta):
        """One step of state and traces at parameter ``theta``; returns ``err, delta, D, phi, W``."""
        cell = self.cell
        self.h, D, phi, W = cell.forward(self.h, x, theta)
        prediction = torch.einsum("bmn,bn->bm", self.R, self.h) + self.c0
        err = prediction - y
        delta = torch.einsum("bmn,bm->bn", self.R, err)
        A = D[:, :, None] * W
        propagation = self.exact_coef[:, None, None] * A + torch.diag_embed(
            self.snap_coef[:, None] * torch.diagonal(A, dim1=1, dim2=2) + self.leak_coef[:, None]
        )
        self.Z = torch.bmm(propagation, self.Z) + cell.immediate_jacobian(D, phi)
        if self.tbptt_window:
            self.jacobian_buffer = [(A, D, phi)] + self.jacobian_buffer[: self.tbptt_window - 1]
        return err, delta, D, phi, W

    @torch.no_grad()
    def _tbptt_update(self, delta):
        """``J_K^T delta`` for the ``K``-step truncated sensitivity ``J_K = B_t + A_t B_{t-1} + ...``
        (``K`` terms), by ``K`` back-products."""
        g = torch.zeros(self.B, self.cell.P, dtype=DTYPE, device=self.device)
        v = delta
        for A, D, phi in self.jacobian_buffer:
            g = g + self.cell.immediate_jacobian_transpose_times(D, phi, v)
            v = torch.einsum("bij,bi->bj", A, v)
        return g

    @torch.no_grad()
    def _uoro_step(self, D, phi, W):
        cell = self.cell
        eps = 1e-12
        nu = torch.sign(torch.randn(self.B, cell.n, generator=self.generator, dtype=DTYPE, device=self.device))
        propagated = D * torch.einsum("bij,bj->bi", W, self.uoro_state)
        nu_B = cell.immediate_jacobian_transpose_times(D, phi, nu)
        r0 = torch.sqrt((self.uoro_parameters.norm(dim=1) + eps) / (propagated.norm(dim=1) + eps))
        r1 = torch.sqrt((nu_B.norm(dim=1) + eps) / (nu.norm(dim=1) + eps))
        self.uoro_state = r0[:, None] * propagated + r1[:, None] * nu
        self.uoro_parameters = self.uoro_parameters / r0[:, None] + nu_B / r1[:, None]

    @torch.no_grad()
    def train(self, num_steps, log_every=0):
        """Run ``num_steps`` online updates.  Returns the block-averaged squared error ``(B, blocks)`` when
        ``log_every`` is positive, ``None`` otherwise.  A lane that diverges is frozen at its initial point."""
        curve = []
        block_error = torch.zeros(self.B, dtype=DTYPE, device=self.device)
        for t in range(num_steps):
            x, y, wrapped = self.stream.step()
            if wrapped:
                self.reset_state()
            err, delta, D, phi, W = self.forward(x, y, self.theta)
            g = torch.einsum("bnp,bn->bp", self.Z, delta)
            if self.uoro_mask.any():
                self._uoro_step(D, phi, W)
                g_uoro = self.uoro_parameters * (self.uoro_state * delta).sum(1, keepdim=True)
                g = torch.where(self.uoro_mask[:, None] > 0, g_uoro, g)
            if self.tbptt_window:
                g = torch.where(self.tbptt_mask[:, None] > 0, self._tbptt_update(delta), g)
            if self.restoration is not None:
                b = self.restoration(t, self.Z, delta, D, phi, W)
                g = g + (b if self.restoration_mask is None else b * self.restoration_mask[:, None])
            if self.clip:
                norm = g.norm(dim=1, keepdim=True)
                g = g * torch.clamp(self.clip / (norm + 1e-300), max=1.0)
            step = self.step_size * self.alive
            readout_step = self.readout_step_size * self.alive
            self.theta = self.theta - step[:, None] * g
            self.R = self.R - readout_step[:, None, None] * err[:, :, None] * self.h[:, None, :]
            self.c0 = self.c0 - readout_step[:, None] * err
            bad = (
                ~torch.isfinite(self.theta).all(1)
                | ~torch.isfinite(self.R).all(dim=(1, 2))
                | (self.theta.abs().amax(1) > DIVERGENCE_BOUND)
            )
            if bad.any():
                self.alive &= ~bad
                self.theta = torch.where(bad[:, None], self.cell.theta, self.theta)
                self.R = torch.where(bad[:, None, None], torch.zeros_like(self.R), self.R)
                self.c0 = torch.where(bad[:, None], torch.zeros_like(self.c0), self.c0)
                self.h = torch.where(bad[:, None], torch.zeros_like(self.h), self.h)
                self.Z[bad] = 0
                self.uoro_state[bad] = 0
                self.uoro_parameters[bad] = 0
                if self.restoration is not None:
                    self.restoration.reset_lanes(bad)
            if log_every:
                block_error += (err * err).sum(1)
                if (t + 1) % log_every == 0:
                    curve.append((block_error / log_every).cpu())
                    block_error.zero_()
        return torch.stack(curve, 1) if curve else None

    @torch.no_grad()
    def evaluate(self, burn_in=200):
        """Frozen-parameter pass over the evaluation series.

        Returns per-lane tensors (NaN on diverged lanes):

        ``mse``                   held-out mean squared error
        ``gradient_norm``         norm of the exact gradient of the recurrent parameters
        ``mean_update_norm``      norm of the mean update of the lane's own rule
        ``bias_field_norm``       norm of their difference, the bias field
        ``gradient_cosine``       cosine between the mean update and the gradient
        ``score_predictability``  predictable share of the error: R^2 of a linear regression of ``e_t`` on
                                  its ten predecessors
        """
        self.reset_state()
        cell = self.cell
        J = torch.zeros(self.B, self.n, cell.P, dtype=DTYPE, device=self.device)
        mse = torch.zeros(self.B, dtype=DTYPE, device=self.device)
        gradient = torch.zeros(self.B, cell.P, dtype=DTYPE, device=self.device)
        mean_update = torch.zeros_like(gradient)
        count = 0
        errors = []
        for i, (x, y) in enumerate(self.stream.eval_iter()):
            err, delta, D, phi, W = self.forward(x, y, self.theta)
            J = torch.bmm(D[:, :, None] * W, J) + cell.immediate_jacobian(D, phi)
            if i >= burn_in:
                update = torch.einsum("bnp,bn->bp", self.Z, delta)
                if self.tbptt_window:
                    update = torch.where(self.tbptt_mask[:, None] > 0, self._tbptt_update(delta), update)
                mse += (err * err).sum(1)
                gradient += torch.einsum("bnp,bn->bp", J, delta)
                mean_update += update
                count += 1
                errors.append(err[:, 0])
        E = torch.stack(errors, 1)
        E = torch.nan_to_num(E - E.mean(1, keepdim=True))
        lags = 10
        X = torch.stack([E[:, lags - j - 1 : E.shape[1] - j - 1] for j in range(lags)], 2)
        Y = E[:, lags:]
        gram = torch.einsum("btj,btk->bjk", X, X) + 1e-10 * torch.eye(lags, dtype=DTYPE, device=self.device)
        moment = torch.einsum("btj,bt->bj", X, Y)
        r2 = (torch.linalg.solve(gram, moment) * moment).sum(1) / (Y * Y).sum(1).clamp_min(1e-300)

        def nan_if_diverged(v):
            return torch.where(self.alive, v, torch.full_like(v, float("nan")))

        # UORO is unbiased: its mean update is the gradient
        mean_update = torch.where(self.uoro_mask[:, None] > 0, gradient, mean_update)
        cosine = (gradient * mean_update).sum(1) / (gradient.norm(dim=1) * mean_update.norm(dim=1) + 1e-300)
        return dict(
            score_predictability=nan_if_diverged(r2),
            gradient_cosine=nan_if_diverged(cosine),
            mse=nan_if_diverged(mse / count),
            gradient_norm=nan_if_diverged((gradient / count).norm(dim=1)),
            mean_update_norm=nan_if_diverged((mean_update / count).norm(dim=1)),
            bias_field_norm=nan_if_diverged(((gradient - mean_update) / count).norm(dim=1)),
        )
