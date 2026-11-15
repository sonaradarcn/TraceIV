"""Recurrent cells with analytic Jacobians.

Everything is dense and lane-batched: a *lane* is an independent run (a system, a seed, a rule, a step size),
so one call advances ``B`` experiments.  Shapes: state ``h`` is ``(B, n)``, parameters ``theta`` are
``(B, P)``, a sensitivity is ``(B, n, P)`` with entry ``[b, k, p] = d h_k / d theta_p``.

Every cell implements the same contract.  ``forward(h, x, theta)`` returns ``(h_new, D, phi, W)`` such that

* the state Jacobian is ``A_t = diag(D) W``,
* the immediate Jacobian ``B_t = d h_t / d theta`` (state held fixed) is ``immediate_jacobian(D, phi)``.

For the tanh cell ``D`` is the derivative of the nonlinearity, ``phi`` the regressor and ``W`` the recurrent
matrix.  The gated cells return ``D = 1`` and put the full state Jacobian in ``W``; ``phi`` then carries the
non-zero part of ``B_t``.
"""

from __future__ import annotations

import torch

from .config import DTYPE

TANH = "tanh"
LINEAR = "linear"
GRU_RESET_AFTER = "gru_reset_after"
GRU_RESET_BEFORE = "gru_reset_before"
GATED_CELLS = (GRU_RESET_AFTER, GRU_RESET_BEFORE)


class RecurrentCell:
    """Elman cell ``h_t = f(W h_{t-1} + V x_t + c)`` with ``f`` the hyperbolic tangent or the identity.

    The parameters ``theta = (W, V, c)`` are stored row-wise per receiving unit: parameter index
    ``p = i * q + a`` with ``q = n + d + 1`` (``n + d`` without bias) and regressor
    ``phi_t = [h_{t-1}; x_t; 1]``.  The immediate Jacobian is ``B_t[k, (i, a)] = 1[k = i] D_k phi_a`` with
    ``D = 1 - h_t^2`` (``D = 1`` for the linear cell).
    """

    def __init__(self, n: int, d: int, B: int, device, nonlinearity: str = TANH, bias: bool = True):
        if nonlinearity not in (TANH, LINEAR) + GATED_CELLS:
            raise ValueError(f"unknown nonlinearity {nonlinearity!r}")
        self.n, self.d, self.B, self.device, self.bias = n, d, B, device, bias
        self.nonlinearity = nonlinearity
        self.q = n + d + (1 if bias else 0)
        self.P = n * self.q
        self.theta = torch.zeros(B, self.P, dtype=DTYPE, device=device)

    # -- parameter views ----------------------------------------------------------------------------
    def weight_matrices(self, theta=None):
        """Views ``(W, V, c)`` of a parameter vector (``c`` is ``None`` without bias)."""
        th = self.theta if theta is None else theta
        M = th.view(-1, self.n, self.q)
        W, V = M[:, :, : self.n], M[:, :, self.n : self.n + self.d]
        c = M[:, :, -1] if self.bias else None
        return W, V, c

    def set_weights(self, W, V, c=None):
        """Write recurrent weights, input weights and (optionally) biases into ``self.theta``."""
        M = self.theta.view(self.B, self.n, self.q)
        M[:, :, : self.n] = W
        M[:, :, self.n : self.n + self.d] = V
        if self.bias:
            M[:, :, -1] = 0.0 if c is None else c

    # -- one step -----------------------------------------------------------------------------------
    def forward(self, h, x, theta=None):
        """One step.  Returns ``h_new``, ``D`` (B, n), ``phi`` (B, q) and ``W`` (B, n, n)."""
        W, V, c = self.weight_matrices(theta)
        a = torch.einsum("bij,bj->bi", W, h) + torch.einsum("bij,bj->bi", V, x)
        if self.bias:
            a = a + c
        parts = [h, x] + ([torch.ones(h.shape[0], 1, dtype=DTYPE, device=h.device)] if self.bias else [])
        phi = torch.cat(parts, dim=1)
        if self.nonlinearity == TANH:
            h_new = torch.tanh(a)
            D = 1 - h_new * h_new
        else:
            h_new = a
            D = torch.ones_like(a)
        return h_new, D, phi, W

    def immediate_jacobian(self, D, phi):
        """``B_t`` as a dense ``(B, n, P)`` tensor."""
        batch = D.shape[0]
        out = torch.zeros(batch, self.n, self.n, self.q, dtype=DTYPE, device=D.device)
        idx = torch.arange(self.n, device=D.device)
        out[:, idx, idx, :] = D[:, :, None] * phi[:, None, :]
        return out.view(batch, self.n, self.P)

    def immediate_jacobian_times(self, D, phi, U):
        """``B_t @ U`` for ``U`` of shape ``(B, P, k)``, without forming ``B_t``: ``(B, n, k)``."""
        Um = U.view(U.shape[0], self.n, self.q, -1)
        return D[:, :, None] * torch.einsum("ba,biak->bik", phi, Um)

    def immediate_jacobian_transpose_times(self, D, phi, nu):
        """``B_t^T nu`` as a parameter vector ``(B, P)``."""
        return ((nu * D)[:, :, None] * phi[:, None, :]).reshape(nu.shape[0], self.P)


class GRUResetAfterCell(RecurrentCell):
    """Gated recurrent unit with the reset gate applied after the recurrent product (cuDNN / PyTorch).

        z = sigm(Wz phi),  r = sigm(Wr phi),  c = tanh(Wn_x [x; 1] + r * (Wn_h h)),  h' = (1 - z) h + z c,

    with ``phi = [h; x; 1]``.  Unit ``i`` owns the three rows ``(Wz_i, Wr_i, Wn_i)``, stored consecutively,
    so the parameter index is ``p = i * q + a`` with ``q = 3 (n + d + bias)`` and the immediate Jacobian is
    block diagonal exactly as for the tanh cell.  ``forward`` honours the contract of the base class with
    ``D = 1`` and ``W = A_t``; the returned ``phi`` has shape ``(B, n, q)`` and holds the non-zero row of
    ``B_t`` of every unit.
    """

    def __init__(self, n: int, d: int, B: int, device, bias: bool = True):
        super().__init__(n, d, B, device, GRU_RESET_AFTER, bias)
        self.gate_fan_in = n + d + (1 if bias else 0)
        self.q = 3 * self.gate_fan_in
        self.P = n * self.q
        self.theta = torch.zeros(B, self.P, dtype=DTYPE, device=device)

    def gate_weights(self, theta=None):
        """Views ``(Wz, Wr, Wn)``, each of shape ``(B, n, n + d + bias)``."""
        th = self.theta if theta is None else theta
        M = th.view(-1, self.n, 3, self.gate_fan_in)
        return M[:, :, 0], M[:, :, 1], M[:, :, 2]

    def weight_matrices(self, theta=None):
        Wn = self.gate_weights(theta)[2]
        return Wn[:, :, : self.n], Wn[:, :, self.n : self.n + self.d], (Wn[:, :, -1] if self.bias else None)

    def set_weights(self, W, V, c=None):
        raise NotImplementedError("assign the parameter vector `theta` of a gated cell directly")

    def forward(self, h, x, theta=None):
        n = self.n
        Wz, Wr, Wn = self.gate_weights(theta)
        parts = [h, x] + ([torch.ones(h.shape[0], 1, dtype=DTYPE, device=h.device)] if self.bias else [])
        phi = torch.cat(parts, dim=1)
        z = torch.sigmoid(torch.einsum("bia,ba->bi", Wz, phi))
        r = torch.sigmoid(torch.einsum("bia,ba->bi", Wr, phi))
        u = torch.einsum("bij,bj->bi", Wn[:, :, :n], h)
        c = torch.tanh(torch.einsum("bia,ba->bi", Wn[:, :, n:], phi[:, n:]) + r * u)
        h_new = (1 - z) * h + z * c
        gz = (c - h) * z * (1 - z)
        gc = z * (1 - c * c)
        gr = gc * u * r * (1 - r)
        psi = phi[:, None, :].expand(-1, n, -1).clone()
        psi[:, :, :n] = r[:, :, None] * h[:, None, :]
        rows = torch.cat(
            [gz[:, :, None] * phi[:, None, :], gr[:, :, None] * phi[:, None, :], gc[:, :, None] * psi], dim=2
        )
        A = (
            torch.diag_embed(1 - z)
            + gz[:, :, None] * Wz[:, :, :n]
            + gr[:, :, None] * Wr[:, :, :n]
            + (gc * r)[:, :, None] * Wn[:, :, :n]
        )
        return h_new, torch.ones_like(h_new), rows, A

    def immediate_jacobian(self, D, phi):
        batch = D.shape[0]
        out = torch.zeros(batch, self.n, self.n, self.q, dtype=DTYPE, device=D.device)
        idx = torch.arange(self.n, device=D.device)
        out[:, idx, idx, :] = phi
        return out.view(batch, self.n, self.P)

    def immediate_jacobian_times(self, D, phi, U):
        Um = U.view(U.shape[0], self.n, self.q, -1)
        return torch.einsum("bia,biak->bik", phi, Um)

    def immediate_jacobian_transpose_times(self, D, phi, nu):
        return (nu[:, :, None] * phi).reshape(nu.shape[0], self.P)


class GRUResetBeforeCell(GRUResetAfterCell):
    """Gated recurrent unit with the reset gate applied before the recurrent product (original form).

        c = tanh(Wn [r * h; x; 1]),   with z, r and h' as in :class:`GRUResetAfterCell`.

    The candidate of unit ``i`` now depends on the reset gates of all units, so the reset-gate weights of
    unit ``j`` enter every row of ``B_t`` and the immediate Jacobian is no longer block diagonal.
    ``support`` (n, P) marks its non-zero pattern; ``forward`` returns the dense ``B_t`` in place of ``phi``.
    """

    def __init__(self, n: int, d: int, B: int, device, bias: bool = True):
        super().__init__(n, d, B, device, bias=bias)
        self.nonlinearity = GRU_RESET_BEFORE
        pattern = torch.zeros(n, n, 3, self.gate_fan_in, dtype=DTYPE, device=device)
        idx = torch.arange(n, device=device)
        pattern[idx, idx] = 1.0
        pattern[:, :, 1] = 1.0
        self.support = pattern.view(n, self.P)

    def forward(self, h, x, theta=None):
        n = self.n
        batch = h.shape[0]
        Wz, Wr, Wn = self.gate_weights(theta)
        parts = [h, x] + ([torch.ones(batch, 1, dtype=DTYPE, device=h.device)] if self.bias else [])
        phi = torch.cat(parts, dim=1)
        z = torch.sigmoid(torch.einsum("bia,ba->bi", Wz, phi))
        r = torch.sigmoid(torch.einsum("bia,ba->bi", Wr, phi))
        psi = phi.clone()
        psi[:, :n] = r * h
        c = torch.tanh(torch.einsum("bia,ba->bi", Wn, psi))
        h_new = (1 - z) * h + z * c
        gz = (c - h) * z * (1 - z)
        gc = z * (1 - c * c)
        # K[b, i, j] = d h'_i / d (pre-activation of r_j)
        K = gc[:, :, None] * Wn[:, :, :n] * (h * r * (1 - r))[:, None, :]
        out = torch.zeros(batch, n, n, 3, self.gate_fan_in, dtype=DTYPE, device=h.device)
        idx = torch.arange(n, device=h.device)
        out[:, idx, idx, 0, :] = gz[:, :, None] * phi[:, None, :]
        out[:, idx, idx, 2, :] = gc[:, :, None] * psi[:, None, :]
        out[:, :, :, 1, :] = K[:, :, :, None] * phi[:, None, None, :]
        A = (
            torch.diag_embed(1 - z)
            + gz[:, :, None] * Wz[:, :, :n]
            + gc[:, :, None] * Wn[:, :, :n] * r[:, None, :]
            + torch.bmm(K, Wr[:, :, :n])
        )
        return h_new, torch.ones_like(h_new), out.view(batch, n, self.P), A

    def immediate_jacobian(self, D, phi):
        return phi

    def immediate_jacobian_times(self, D, phi, U):
        return torch.bmm(phi, U)

    def immediate_jacobian_transpose_times(self, D, phi, nu):
        return torch.einsum("bnp,bn->bp", phi, nu)


def make_cell(n: int, d: int, B: int, device, kind: str = TANH, bias: bool = True) -> RecurrentCell:
    """Cell of the given kind: ``tanh``, ``linear``, ``gru_reset_after`` or ``gru_reset_before``."""
    if kind == GRU_RESET_AFTER:
        return GRUResetAfterCell(n, d, B, device, bias=bias)
    if kind == GRU_RESET_BEFORE:
        return GRUResetBeforeCell(n, d, B, device, bias=bias)
    return RecurrentCell(n, d, B, device, kind, bias)
