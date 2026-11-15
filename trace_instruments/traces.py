"""Causal traces: the approximate sensitivities that online rules propagate in place of ``J``.

All deterministic rules share one recursion,

    Z_t = L_t Z_{t-1} + B_t,

and differ in the propagation operator ``L_t``: the state Jacobian ``A_t`` (RTRL), its restriction to the
support of the immediate Jacobian (SnAp-1), zero (immediate rule) or a multiple of the identity (leak).
"""

from __future__ import annotations

import torch

from . import rules
from .cells import RecurrentCell
from .config import DTYPE


def snap1_propagate(cell: RecurrentCell, A, Z):
    """SnAp-1 propagation: the product ``A Z`` restricted to the support of the immediate Jacobian.

    For a cell whose parameters are owned by single units that restriction is ``diag(A) Z``.
    """
    support = getattr(cell, "support", None)
    if support is None:
        return torch.diagonal(A, dim1=1, dim2=2)[:, :, None] * Z
    return support * torch.bmm(A, Z)


class TraceBank:
    """A set of named traces advanced together on the same trajectory.

    ``rule_names`` may contain ``rtrl``, ``snap1``, ``immediate``, ``leak:<rate>``, ``tbptt:<K>`` and
    ``hybrid:<m>`` (see :mod:`trace_instruments.rules`).  After :meth:`step`, ``self.Z[name]`` is the
    current trace of each rule, of shape ``(B, n, P)``.
    """

    def __init__(self, cell: RecurrentCell, rule_names, B=None):
        self.cell, self.rule_names = cell, list(rule_names)
        B = cell.B if B is None else B

        def zeros():
            return torch.zeros(B, cell.n, cell.P, dtype=DTYPE, device=cell.device)

        self.Z = {}
        self.parts = {}  # auxiliary tensors of the rules that need more than one
        self.exact_columns = {}
        for name in self.rule_names:
            self.Z[name] = zeros()
            if rules.is_tbptt(name):
                self.parts[name] = [zeros() for _ in range(rules.tbptt_window(name))]
            if rules.is_hybrid(name):
                self.parts[name] = [zeros(), zeros()]  # SnAp-1 part, exact part
                columns = torch.zeros(cell.n, cell.q, dtype=DTYPE, device=cell.device)
                columns[rules.hybrid_split(name) :] = 1.0
                self.exact_columns[name] = columns.view(1, 1, cell.P)

    def step(self, D, phi, W):
        """Advance every trace by one step.  Returns the immediate Jacobian and the state Jacobian."""
        cell = self.cell
        Bt = cell.immediate_jacobian(D, phi)
        A = D[:, :, None] * W
        for name in self.rule_names:
            if name == rules.RTRL:
                self.Z[name] = torch.bmm(A, self.Z[name]) + Bt
            elif name == rules.SNAP1:
                self.Z[name] = snap1_propagate(cell, A, self.Z[name]) + Bt
            elif name == rules.IMMEDIATE:
                self.Z[name] = Bt
            elif rules.is_leak(name):
                self.Z[name] = rules.leak_rate(name) * self.Z[name] + Bt
            elif rules.is_hybrid(name):
                diag_A = torch.diagonal(A, dim1=1, dim2=2)
                exact = self.exact_columns[name]
                self.parts[name] = [
                    diag_A[:, :, None] * self.parts[name][0] + Bt * (1 - exact),
                    torch.bmm(A, self.parts[name][1]) + Bt * exact,
                ]
                self.Z[name] = self.parts[name][0] + self.parts[name][1]
            elif rules.is_tbptt(name):
                truncated = self.parts[name]  # truncated[j] = (j + 1)-step truncated sensitivity
                new = [Bt]
                for j in range(1, len(truncated)):
                    new.append(torch.bmm(A, truncated[j - 1]) + Bt)
                self.parts[name] = new
                self.Z[name] = new[-1]
            else:
                raise ValueError(name)
        return Bt, A

    def reset(self):
        """Zero all traces."""
        for name in self.Z:
            self.Z[name].zero_()
        for name in self.parts:
            for tensor in self.parts[name]:
                tensor.zero_()


def lane_coefficients(rule_names, device):
    """Per-lane coefficients ``(a, c, l)`` of the unified recursion
    ``Z <- (a A + diag(c diag(A) + l)) Z + B_t``, one entry per lane."""
    for name in rule_names:
        known = name in (rules.RTRL, rules.SNAP1, rules.IMMEDIATE) or rules.is_leak(name) or rules.is_hybrid(name)
        assert known, name
    a = torch.tensor([1.0 if name == rules.RTRL else 0.0 for name in rule_names], dtype=DTYPE, device=device)
    c = torch.tensor(
        [1.0 if name == rules.SNAP1 or rules.is_hybrid(name) else 0.0 for name in rule_names],
        dtype=DTYPE,
        device=device,
    )
    leak = torch.tensor(
        [rules.leak_rate(name) if rules.is_leak(name) else 0.0 for name in rule_names], dtype=DTYPE, device=device
    )
    return a, c, leak


def hybrid_exact_columns(rule_names, cell: RecurrentCell, device):
    """``(B, P)`` mask of the parameter columns that a lane propagates exactly.

    Rule ``hybrid:<m>`` keeps the SnAp-1 trace for the incoming weights of units ``0..m-1`` and the exact
    sensitivity for those of units ``m..n-1``; all other rules have an empty mask.
    """
    out = torch.zeros(len(rule_names), cell.n, cell.q, dtype=DTYPE, device=device)
    for b, name in enumerate(rule_names):
        if rules.is_hybrid(name):
            out[b, rules.hybrid_split(name) :] = 1.0
    return out.view(len(rule_names), cell.P)
