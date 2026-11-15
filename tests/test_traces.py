"""The trace recursions: RTRL against automatic differentiation, and the limiting cases of the other rules."""

import pytest
import torch

from trace_instruments import DTYPE, rules
from trace_instruments.cells import GRU_RESET_AFTER, GRU_RESET_BEFORE, TANH, make_cell
from trace_instruments.traces import TraceBank, hybrid_exact_columns, lane_coefficients

N, D, B, T = 3, 2, 2, 6


def roll_out(kind, rule_names, seed=0):
    """Run ``T`` steps from a zero state; returns the cell, the parameters, the inputs and the trace bank."""
    generator = torch.Generator().manual_seed(seed)
    cell = make_cell(N, D, B, "cpu", kind)
    theta = 0.5 * torch.randn(B, cell.P, generator=generator, dtype=DTYPE)
    xs = torch.randn(T, B, D, generator=generator, dtype=DTYPE)
    bank = TraceBank(cell, rule_names)
    h = torch.zeros(B, N, dtype=DTYPE)
    for t in range(T):
        h, D_t, phi, W = cell.forward(h, xs[t], theta)
        bank.step(D_t, phi, W)
    return cell, theta, xs, bank


@pytest.mark.parametrize("kind", [TANH, GRU_RESET_AFTER, GRU_RESET_BEFORE])
def test_rtrl_trace_is_the_exact_sensitivity(kind):
    cell, theta, xs, bank = roll_out(kind, [rules.RTRL])
    for b in range(B):

        def final_state(theta_b):
            h = torch.zeros(1, N, dtype=DTYPE)
            for t in range(T):
                h = cell.forward(h, xs[t, b : b + 1], theta_b[None])[0]
            return h[0]

        expected = torch.autograd.functional.jacobian(final_state, theta[b])
        torch.testing.assert_close(bank.Z[rules.RTRL][b], expected, rtol=1e-9, atol=1e-12)


def test_limiting_cases_of_the_rules():
    names = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.0", f"tbptt:{T}", "tbptt:1", "hybrid:0", f"hybrid:{N}"]
    _, _, _, bank = roll_out(TANH, names)
    Z = bank.Z
    torch.testing.assert_close(Z["leak:0.0"], Z[rules.IMMEDIATE])  # no memory
    torch.testing.assert_close(Z["tbptt:1"], Z[rules.IMMEDIATE])  # window of one step
    torch.testing.assert_close(Z[f"tbptt:{T}"], Z[rules.RTRL])  # window as long as the run
    torch.testing.assert_close(Z["hybrid:0"], Z[rules.RTRL])  # every unit exact
    torch.testing.assert_close(Z[f"hybrid:{N}"], Z[rules.SNAP1])  # no unit exact


def test_snap1_keeps_the_support_of_the_immediate_jacobian():
    cell, _, _, bank = roll_out(TANH, [rules.RTRL, rules.SNAP1])
    off_diagonal = ~torch.eye(N, dtype=torch.bool)
    assert bank.Z[rules.SNAP1].view(B, N, N, cell.q)[:, off_diagonal].abs().max() == 0
    assert bank.Z[rules.RTRL].view(B, N, N, cell.q)[:, off_diagonal].abs().max() > 0


def test_lane_coefficients_and_hybrid_columns():
    names = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5", "hybrid:1"]
    a, c, leak = lane_coefficients(names, "cpu")
    assert a.tolist() == [1, 0, 0, 0, 0]
    assert c.tolist() == [0, 1, 0, 0, 1]
    assert leak.tolist() == [0, 0, 0, 0.5, 0]
    cell = make_cell(N, D, len(names), "cpu", TANH)
    columns = hybrid_exact_columns(names, cell, "cpu").view(len(names), N, cell.q)
    assert columns[:4].sum() == 0
    assert columns[4, 0].sum() == 0 and columns[4, 1:].min() == 1


def test_rule_name_parsing():
    assert rules.leak_rate("leak:0.9") == 0.9
    assert rules.tbptt_window("tbptt:8") == 8
    assert rules.hybrid_split("hybrid:3") == 3
    assert rules.is_leak("leak:0.5") and not rules.is_leak(rules.SNAP1)
