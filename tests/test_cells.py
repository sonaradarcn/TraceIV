"""The analytic Jacobians of every cell against automatic differentiation."""

import pytest
import torch

from trace_instruments import DTYPE
from trace_instruments.cells import GRU_RESET_AFTER, GRU_RESET_BEFORE, LINEAR, TANH, make_cell

KINDS = [TANH, LINEAR, GRU_RESET_AFTER, GRU_RESET_BEFORE]
N, D, B = 4, 2, 3


def random_step(kind, bias, seed=0):
    generator = torch.Generator().manual_seed(seed)
    cell = make_cell(N, D, B, "cpu", kind, bias)
    theta = 0.5 * torch.randn(B, cell.P, generator=generator, dtype=DTYPE)
    h = torch.randn(B, N, generator=generator, dtype=DTYPE).tanh()
    x = torch.randn(B, D, generator=generator, dtype=DTYPE)
    return cell, theta, h, x


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bias", [True, False])
def test_jacobians_match_autograd(kind, bias):
    cell, theta, h, x = random_step(kind, bias)
    _, D, phi, W = cell.forward(h, x, theta)
    state_jacobian = D[:, :, None] * W
    immediate_jacobian = cell.immediate_jacobian(D, phi)
    for b in range(B):

        def step(h_b, theta_b):
            return cell.forward(h_b[None], x[b : b + 1], theta_b[None])[0][0]

        d_state, d_theta = torch.autograd.functional.jacobian(step, (h[b], theta[b]))
        torch.testing.assert_close(state_jacobian[b], d_state, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(immediate_jacobian[b], d_theta, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_jacobian_products_match_dense_jacobian(kind):
    cell, theta, h, x = random_step(kind, bias=True, seed=1)
    _, D, phi, _ = cell.forward(h, x, theta)
    dense = cell.immediate_jacobian(D, phi)
    generator = torch.Generator().manual_seed(2)
    U = torch.randn(B, cell.P, 3, generator=generator, dtype=DTYPE)
    nu = torch.randn(B, N, generator=generator, dtype=DTYPE)
    torch.testing.assert_close(cell.immediate_jacobian_times(D, phi, U), torch.bmm(dense, U))
    torch.testing.assert_close(
        cell.immediate_jacobian_transpose_times(D, phi, nu), torch.einsum("bnp,bn->bp", dense, nu)
    )


@pytest.mark.parametrize("kind", [TANH, GRU_RESET_AFTER])
def test_immediate_jacobian_is_block_diagonal(kind):
    """Unit ``i`` owns its incoming weights: row ``i`` of ``B_t`` is zero outside block ``i``."""
    cell, theta, h, x = random_step(kind, bias=True)
    _, D, phi, _ = cell.forward(h, x, theta)
    blocks = cell.immediate_jacobian(D, phi).view(B, N, N, cell.q)
    off_diagonal = ~torch.eye(N, dtype=torch.bool)
    assert blocks[:, off_diagonal].abs().max() == 0


def test_reset_before_support_covers_immediate_jacobian():
    cell, theta, h, x = random_step(GRU_RESET_BEFORE, bias=True)
    _, D, phi, _ = cell.forward(h, x, theta)
    dense = cell.immediate_jacobian(D, phi)
    assert (dense * (1 - cell.support)).abs().max() == 0
    assert cell.support.sum() > N * cell.q  # more than the block diagonal: the reset gates couple the units
