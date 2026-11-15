"""Teacher-student systems: a recurrent teacher generates the targets a recurrent student is trained on.

Lanes.  A :class:`TeacherStudentSystem` holds ``S`` independent systems, each replicated; the replicas of
one system see the same inputs and noise (common random numbers) and may run different learning rules,
step sizes or restoration settings.  Everything is advanced by one batched call per time step.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .cells import GATED_CELLS, GRU_RESET_AFTER, TANH, GRUResetAfterCell, GRUResetBeforeCell, RecurrentCell
from .config import DTYPE

# Structural families of recurrent matrices (see `sample_recurrent_matrices`).
NORMAL = "normal"
ORTHOGONAL = "orthogonal"
RANDOM = "random"
CHAIN = "chain"
SCHUR_PREFIX = "schur:"
SCHUR_FIXED_PREFIX = "schur_fixed:"


def sample_recurrent_matrices(n, family, gain, generator, count):
    """Recurrent matrices ``(count, n, n)`` of a structural family.

    ``normal``                   symmetric: real spectrum, numerical radius equal to the spectral radius
    ``orthogonal``               scaled orthogonal: normal, complex spectrum
    ``random``                   Ginibre, entries of variance ``gain^2 / n``
    ``chain``                    feed-forward chain: small spectral radius, numerical radius about ``gain``
    ``schur:<s>``                Schur form with a fixed spectrum and the strict upper part scaled by ``s``
    ``schur_fixed:<seed>:<s>``   one fixed Schur-form matrix drawn with NumPy from ``seed`` (the family of
                                 the linear stability experiment); ``gain`` is not used
    """
    out = []
    for _ in range(count):
        g = torch.randn(n, n, generator=generator, dtype=DTYPE)
        if family == NORMAL:
            Q, _ = torch.linalg.qr(g)
            lam = (0.3 + 0.7 * torch.rand(n, generator=generator, dtype=DTYPE)) * torch.where(
                torch.rand(n, generator=generator) < 0.5, -1.0, 1.0
            )
            W = gain * Q @ torch.diag(lam) @ Q.T
        elif family == ORTHOGONAL:
            Q, _ = torch.linalg.qr(g)
            W = gain * Q
        elif family == RANDOM:
            W = gain * g / math.sqrt(n)
        elif family == CHAIN:
            W = 0.3 * torch.eye(n, dtype=DTYPE) + gain * torch.diag(torch.ones(n - 1, dtype=DTYPE), 1)
        elif family.startswith(SCHUR_PREFIX):
            upper_scale = float(family.split(":")[1])
            Q, _ = torch.linalg.qr(g)
            lam = (
                gain
                * (0.3 + 0.7 * torch.rand(n, generator=generator, dtype=DTYPE))
                * torch.where(torch.rand(n, generator=generator) < 0.5, -1.0, 1.0)
            )
            T = torch.diag(lam) + upper_scale * torch.triu(
                torch.randn(n, n, generator=generator, dtype=DTYPE), 1
            ) / math.sqrt(n)
            W = Q @ T @ Q.T
        elif family.startswith(SCHUR_FIXED_PREFIX):
            _, numpy_seed, upper_scale = family.split(":")
            rng = np.random.default_rng(int(numpy_seed))
            Qn, _ = np.linalg.qr(rng.standard_normal((n, n)))
            lam = rng.uniform(0.3, 0.8, n) * rng.choice([-1, 1], n)
            Nn = np.triu(rng.standard_normal((n, n)), 1) / np.sqrt(n)
            W = torch.tensor(Qn @ (np.diag(lam) + float(upper_scale) * Nn) @ Qn.T, dtype=DTYPE)
        else:
            raise ValueError(family)
        out.append(W)
    return torch.stack(out)


class TeacherStudentSystem:
    """Teacher with ``teacher_units`` units, student with ``student_units``; the target is the teacher
    read-out plus noise, and the student is read out through the matching matrix.

    Parameters
    ----------
    teacher_units, student_units, input_dim
        Sizes.  A student smaller than the teacher is misspecified.
    num_systems
        Number ``S`` of independent systems.
    replicas
        Lanes per data stream; replicas share inputs and noise.
    independent_streams
        Independent data streams per system.
    family, gain
        Structural family and scale of the teacher's recurrent matrix.
    num_readout, readout
        ``units``: the read-out selects the first ``num_readout`` units.  ``dense``: a fixed random
        ``num_readout x n`` matrix (the same matrix reads teacher and student, so the problem stays well
        specified when the sizes agree).  ``num_readout=None`` reads every student unit.
    noise_std, noise_ar
        Observation noise: white ``N(0, noise_std^2)``, or AR(1) with coefficient ``noise_ar``.
    cell
        ``tanh``, ``linear``, ``gru_reset_after`` or ``gru_reset_before``.
    feed_back_target
        Feed the previous target back as an additional student input.
    input_scale
        Scale of the input weights.
    coupling
        Strength with which the unmodelled teacher units drive the modelled ones (misspecification).

    Attributes
    ----------
    teacher, student : cells
    initial_parameters : (B, P) student parameters at construction; they equal the teacher's (the
        calibrated point ``theta*``) when the sizes agree and the target is not fed back.
    readout_gram : (B, n, n) matrix ``F = R^T R`` of the student read-out.
    generator : random generator of the data stream (reseed it to draw a different stream).
    """

    def __init__(
        self,
        teacher_units,
        student_units,
        input_dim,
        num_systems,
        device,
        replicas=1,
        independent_streams=1,
        family=RANDOM,
        gain=0.9,
        num_readout=None,
        noise_std=0.1,
        seed=0,
        cell=TANH,
        noise_ar=0.0,
        feed_back_target=False,
        input_scale=1.0,
        readout="units",
        coupling=None,
    ):
        n_t, n_s, d, S = teacher_units, student_units, input_dim, num_systems
        generator = torch.Generator().manual_seed(seed)
        self.device, self.S, self.replicas = device, S, replicas
        self.B = S * independent_streams * replicas
        self.num_streams = S * independent_streams
        self.d, self.noise_std, self.noise_ar, self.feed_back_target = d, noise_std, noise_ar, feed_back_target
        self.m = num_readout or n_s

        def to_lanes(x):
            return x.repeat_interleave(independent_streams * replicas, dim=0).to(device)

        if cell in GATED_CELLS:  # gated teacher and student of the same size; theta* is the teacher
            assert n_t == n_s and not feed_back_target and coupling is None
            cell_class = GRUResetAfterCell if cell == GRU_RESET_AFTER else GRUResetBeforeCell
            self.teacher = cell_class(n_t, d, self.B, device, bias=False)
            self.student = cell_class(n_s, d, self.B, device, bias=False)
            theta = torch.zeros(S, n_t, 3, n_t + d, dtype=DTYPE)
            for gate in range(3):
                theta[:, :, gate, :n_t] = sample_recurrent_matrices(n_t, family, gain, generator, S)
                theta[:, :, gate, n_t:] = (
                    input_scale * torch.randn(S, n_t, d, generator=generator, dtype=DTYPE) / math.sqrt(d)
                )
            self.teacher.theta = to_lanes(theta.reshape(S, -1))
            self.student.theta = self.teacher.theta.clone()
            self.initial_parameters = self.student.theta.clone()
        else:
            self._init_elman(n_t, n_s, d, S, device, cell, family, gain, generator, input_scale, coupling, to_lanes)
        if readout == "units":
            teacher_readout = torch.eye(n_t, dtype=DTYPE)[: self.m].expand(S, -1, -1).clone()
        else:
            teacher_readout = torch.randn(S, self.m, n_t, generator=generator, dtype=DTYPE) / math.sqrt(self.m)
        self.teacher_readout = to_lanes(teacher_readout)
        self.student_readout = to_lanes(teacher_readout[:, :, :n_s])
        self.readout_gram = torch.bmm(self.student_readout.transpose(1, 2), self.student_readout)
        self.generator = torch.Generator(device=device).manual_seed(seed + 12345)
        self.reset_state()

    def _init_elman(self, n_t, n_s, d, S, device, cell, family, gain, generator, input_scale, coupling, to_lanes):
        self.teacher = RecurrentCell(n_t, d, self.B, device, cell, bias=False)
        W_teacher = sample_recurrent_matrices(n_t, family, gain, generator, S)
        if coupling is not None and n_t > n_s:
            W_teacher[:, :n_s, n_s:] *= coupling
        V_teacher = input_scale * torch.randn(S, n_t, d, generator=generator, dtype=DTYPE) / math.sqrt(d)
        self.teacher.set_weights(to_lanes(W_teacher), to_lanes(V_teacher))
        student_input_dim = d + (self.m if self.feed_back_target else 0)
        self.student = RecurrentCell(n_s, student_input_dim, self.B, device, cell, bias=False)
        V_student = torch.zeros(S, n_s, student_input_dim, dtype=DTYPE)
        V_student[:, :, :d] = V_teacher[:, :n_s]
        self.student.set_weights(to_lanes(W_teacher[:, :n_s, :n_s]), to_lanes(V_student))
        self.initial_parameters = self.student.theta.clone()

    def reset_state(self):
        """Zero the states of teacher and student and the noise memory."""

        def zeros(k):
            return torch.zeros(self.B, k, dtype=DTYPE, device=self.device)

        self.teacher_state, self.student_state = zeros(self.teacher.n), zeros(self.student.n)
        self.noise_state, self.previous_target = zeros(self.m), zeros(self.m)

    def randn(self, k):
        """``(B, k)`` normal draws shared by the replicas of a data stream."""
        draws = torch.randn(self.num_streams, k, generator=self.generator, dtype=DTYPE, device=self.device)
        return draws.repeat_interleave(self.replicas, dim=0)

    def step(self, theta=None):
        """Advance teacher and student one step; returns ``(delta, D, phi, W, err)`` for the student."""
        x = self.randn(self.d)
        self.teacher_state, _, _, _ = self.teacher.forward(self.teacher_state, x)
        self.noise_state = self.noise_ar * self.noise_state + self.noise_std * self.randn(self.m)
        y = torch.einsum("bmn,bn->bm", self.teacher_readout, self.teacher_state) + self.noise_state
        student_input = torch.cat([x, self.previous_target], dim=1) if self.feed_back_target else x
        self.student_state, D, phi, W = self.student.forward(self.student_state, student_input, theta)
        err = torch.einsum("bmn,bn->bm", self.student_readout, self.student_state) - y
        delta = torch.einsum("bmn,bm->bn", self.student_readout, err)
        self.previous_target = y
        return delta, D, phi, W, err
