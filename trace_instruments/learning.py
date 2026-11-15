"""Online learning on teacher-student systems, and evaluation of the risk at fixed parameters."""

from __future__ import annotations

import torch

from .config import DIVERGENCE_BOUND, DTYPE
from .teacher_student import TeacherStudentSystem
from .traces import hybrid_exact_columns, lane_coefficients, snap1_propagate


@torch.no_grad()
def run_online_learning(
    system: TeacherStudentSystem,
    rule_names,
    step_size,
    num_steps,
    initial_parameters=None,
    average_from=0.5,
    optimizer="sgd",
    preconditioner=None,
    record_covariance=True,
    restoration=None,
    restoration_mask=None,
    clip=None,
    log_every=0,
):
    """Run ``theta <- theta - step_size * Gamma (Z^T delta + b)`` on all lanes.

    Parameters
    ----------
    rule_names
        One rule name, or a list with one rule per lane (``rtrl``, ``snap1``, ``immediate``, ``leak:<rate>``,
        ``hybrid:<m>``).
    step_size
        Scalar or ``(B,)`` tensor.
    initial_parameters
        Starting point; the system's own initial parameters by default.
    average_from
        Fraction of the run after which the iterate is averaged.
    optimizer, preconditioner
        ``"adam"``, or plain updates multiplied by ``preconditioner`` (B, P, P) when one is given.
    restoration, restoration_mask
        Optional bias-field estimator (:mod:`trace_instruments.restoration`); the mask (B,) selects the
        lanes that add its correction.
    clip
        Optional bound on the norm of the update direction.
    log_every
        If positive, record the mean loss over consecutive blocks of this many steps.

    Returns
    -------
    dict with ``averaged_parameters``, ``final_parameters``, ``covariance`` (of the iterate over the
    averaging window, or ``None``), ``mean_loss`` (over that window), ``diverged`` (per-lane flag) and
    ``loss_curve`` (or ``None``).  A lane that diverges is frozen at the initial parameters of the system.
    """
    student = system.student
    B, P, device = system.B, student.P, system.device
    if isinstance(rule_names, str):
        rule_names = [rule_names] * B
    exact_coef, snap_coef, leak_coef = lane_coefficients(rule_names, device)
    exact_columns = hybrid_exact_columns(rule_names, student, device)
    Z_exact = torch.zeros(B, student.n, P, dtype=DTYPE, device=device) if exact_columns.any() else None
    theta = (system.initial_parameters if initial_parameters is None else initial_parameters).clone()
    Z = torch.zeros(B, student.n, P, dtype=DTYPE, device=device)
    system.reset_state()
    step_size = torch.as_tensor(step_size, dtype=DTYPE, device=device).expand(B).clone()
    sum_theta = torch.zeros(B, P, dtype=DTYPE, device=device)
    sum_outer = torch.zeros(B, P, P, dtype=DTYPE, device=device) if record_covariance else None
    loss = torch.zeros(B, dtype=DTYPE, device=device)
    count = 0
    alive = torch.ones(B, dtype=torch.bool, device=device)
    first_moment = torch.zeros_like(theta)
    second_moment = torch.zeros_like(theta)
    start = int(average_from * num_steps)
    curve = []
    block_loss = torch.zeros(B, dtype=DTYPE, device=device)
    for t in range(num_steps):
        delta, D, phi, W, err = system.step(theta)
        A = D[:, :, None] * W
        propagation = exact_coef[:, None, None] * A + torch.diag_embed(
            snap_coef[:, None] * torch.diagonal(A, dim1=1, dim2=2) + leak_coef[:, None]
        )
        Bt = student.immediate_jacobian(D, phi)
        if getattr(student, "support", None) is not None:
            # SnAp-1 with a support that is not block diagonal
            Z = (
                torch.bmm(exact_coef[:, None, None] * A + torch.diag_embed(leak_coef[:, None].expand(-1, student.n)), Z)
                + snap_coef[:, None, None] * snap1_propagate(student, A, Z)
                + Bt
            )
            Z_full = Z
        elif Z_exact is None:
            Z = torch.bmm(propagation, Z) + Bt
            Z_full = Z
        else:
            # hybrid lanes: exact recursion on the selected columns
            Z = torch.bmm(propagation, Z) + Bt * (1 - exact_columns[:, None, :])
            Z_exact = torch.bmm(A, Z_exact) + Bt * exact_columns[:, None, :]
            Z_full = Z + Z_exact
        g = torch.einsum("bkp,bk->bp", Z_full, delta)
        if restoration is not None:
            b = restoration(t, Z_full, delta, D, phi, W)
            g = g + (b if restoration_mask is None else b * restoration_mask[:, None])
        if clip is not None:
            norm = g.norm(dim=1, keepdim=True)
            g = g * torch.clamp(clip / (norm + 1e-300), max=1.0)
        if optimizer == "adam":
            first_moment = 0.9 * first_moment + 0.1 * g
            second_moment = 0.999 * second_moment + 0.001 * g * g
            update = (first_moment / (1 - 0.9 ** (t + 1))) / ((second_moment / (1 - 0.999 ** (t + 1))).sqrt() + 1e-8)
        elif preconditioner is not None:
            update = torch.einsum("bpq,bq->bp", preconditioner, g)
        else:
            update = g
        theta = theta - step_size[:, None] * update * alive[:, None]
        bad = (
            ~torch.isfinite(theta).all(1)
            | (theta.abs().amax(1) > DIVERGENCE_BOUND)
            | ~torch.isfinite(system.student_state).all(1)
        )
        if bad.any():
            alive &= ~bad
            theta = torch.where(bad[:, None], system.initial_parameters, theta)
            system.student_state = torch.where(
                bad[:, None], torch.zeros_like(system.student_state), system.student_state
            )
            Z[bad] = 0
            if Z_exact is not None:
                Z_exact[bad] = 0
            if restoration is not None:
                restoration.reset_lanes(bad)
        step_loss = 0.5 * (err * err).sum(1)
        if log_every:
            block_loss += step_loss
            if (t + 1) % log_every == 0:
                curve.append((block_loss / log_every).cpu())
                block_loss.zero_()
        if t >= start:
            sum_theta += theta
            count += 1
            loss += step_loss
            if record_covariance:
                sum_outer += theta[:, :, None] * theta[:, None, :]
    mean = sum_theta / count
    covariance = (sum_outer / count - mean[:, :, None] * mean[:, None, :]) if record_covariance else None
    return dict(
        averaged_parameters=mean,
        final_parameters=theta,
        covariance=covariance,
        mean_loss=loss / count,
        diverged=~alive,
        loss_curve=torch.stack(curve, 1) if curve else None,
    )


@torch.no_grad()
def evaluate_risk(system: TeacherStudentSystem, theta, num_steps, burn_in=500):
    """Mean prediction loss (noise included) and exact mean gradient at a fixed parameter."""
    Z = torch.zeros(system.B, system.student.n, system.student.P, dtype=DTYPE, device=system.device)
    system.reset_state()
    loss = torch.zeros(system.B, dtype=DTYPE, device=system.device)
    gradient = torch.zeros(system.B, system.student.P, dtype=DTYPE, device=system.device)
    for t in range(num_steps + burn_in):
        delta, D, phi, W, err = system.step(theta)
        Z = torch.bmm(D[:, :, None] * W, Z) + system.student.immediate_jacobian(D, phi)
        if t >= burn_in:
            loss += 0.5 * (err * err).sum(1)
            gradient += torch.einsum("bkp,bk->bp", Z, delta)
    return loss / num_steps, gradient / num_steps
