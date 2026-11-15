"""The exchange rate of tangent probing (proposition on the exchange rate).

The parameter is frozen at a perturbed point of a teacher-student problem with half of the units read out.
The bias field ``D = E[(J - Z)^T delta]`` of SnAp-1 is first measured with the exact sensitivity; then the
projected update ``b <- b + beta U (s_hat - U^T b)`` with ``k`` rotating exact tangents is run from
``b = 0`` and compared with

    E||b_j - D||^2 = (1 - (2 beta - beta^2) k / P)^j ||D||^2 + [1 - (...)^j] * beta / (2 - beta) * P vs^2 / w,

where ``vs^2 / w`` is the measured variance of a window statistic about its conditional mean.

    python -m experiments.restoration.exchange_rate --num-units 4,10,22
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import float_list, int_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE
from trace_instruments.restoration import PROJECTED, BiasFieldEstimator
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("restoration", "exchange_rate")
NOISE_STD = 0.1


@torch.no_grad()
def measure_bias_field(system, theta, num_steps, burn_in=500):
    """Bias field of SnAp-1 at the fixed parameter ``theta``, from the exact sensitivity."""
    B, n, P = system.B, system.student.n, system.student.P
    J = torch.zeros(B, n, P, dtype=DTYPE, device=system.device)
    Z = torch.zeros_like(J)
    D_field = torch.zeros(B, P, dtype=DTYPE, device=system.device)
    system.reset_state()
    for t in range(num_steps + burn_in):
        delta, D, phi, W, _ = system.step(theta)
        A = D[:, :, None] * W
        Bt = system.student.immediate_jacobian(D, phi)
        J = torch.bmm(A, J) + Bt
        Z = torch.diagonal(A, dim1=1, dim2=2)[:, :, None] * Z + Bt
        if t >= burn_in:
            D_field += torch.einsum("bnp,bn->bp", J - Z, delta)
    return D_field / num_steps


@torch.no_grad()
def track_bias_field(system, theta, D_field, num_tangents, rate, window, num_windows, seed):
    """Run the projected estimator from zero.

    Returns the squared error ``|b - D|^2`` after every window, ``(B, num_windows)``, and the residual of
    every window statistic about ``U^T D``, ``(B, num_windows - 1, num_tangents)``.
    """
    B, n, P = system.B, system.student.n, system.student.P
    estimator = BiasFieldEstimator(
        system.student, B, system.device, mode=PROJECTED, num_tangents=num_tangents, window=window, rate=rate, seed=seed
    )
    Z = torch.zeros(B, n, P, dtype=DTYPE, device=system.device)
    system.reset_state()
    squared_error = []
    residual = []
    for t in range(window * num_windows + 1):
        delta, D, phi, W, _ = system.step(theta)
        A = D[:, :, None] * W
        Z = torch.diagonal(A, dim1=1, dim2=2)[:, :, None] * Z + system.student.immediate_jacobian(D, phi)
        if t > 0 and t % window == 0 and estimator.window_count:  # the statistic about to be folded in
            s = estimator.window_sum / estimator.window_count
            projected_field = torch.bmm(estimator.directions.transpose(1, 2), D_field[:, :, None])[:, :, 0]
            residual.append((s - projected_field).cpu())
        estimator(t, Z, delta, D, phi, W)
        if t > 0 and t % window == 0:
            squared_error.append(((estimator.estimate - D_field) ** 2).sum(1).cpu())
    return torch.stack(squared_error, 1).numpy(), torch.stack(residual, 1).numpy()


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR, device="cpu")
    parser.add_argument("--num-units", default="4,10,22", help="comma-separated network sizes")
    parser.add_argument("--num-systems", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-tangents", default="1,4,16")
    parser.add_argument("--tracking-rates", default="0.05,0.2,1.0")
    parser.add_argument("--windows", default="50,200")
    parser.add_argument("--num-windows", type=int, default=600)
    parser.add_argument("--bias-field-steps", type=int, default=200_000, help="steps of the reference measurement")
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for n in int_list(args.num_units):

        def make_system():
            return TeacherStudentSystem(
                n,
                n,
                args.input_dim,
                args.num_systems,
                args.device,
                family="random",
                gain=args.gain,
                noise_std=NOISE_STD,
                seed=args.seed,
                num_readout=n // 2,
            )

        system = make_system()
        P = system.student.P
        generator = torch.Generator().manual_seed(args.seed + 7)
        direction = torch.randn(args.num_systems, P, generator=generator, dtype=DTYPE)
        direction = direction / direction.norm(dim=1, keepdim=True)
        theta = system.initial_parameters + args.perturbation * system.initial_parameters.norm(
            dim=1, keepdim=True
        ) * direction.to(args.device)
        start = time.time()
        D_field = measure_bias_field(system, theta, args.bias_field_steps)
        print(f"n={n} P={P} |D| median={D_field.norm(dim=1).median():.3e}  [{time.time() - start:.0f}s]", flush=True)
        rows = []
        for k in int_list(args.num_tangents):
            if k > P:
                continue
            for rate in float_list(args.tracking_rates):
                for window in int_list(args.windows):
                    system = make_system()
                    system.generator.manual_seed(999 + k + window)
                    squared_error, residual = track_bias_field(
                        system, theta, D_field, k, rate, window, args.num_windows, args.seed
                    )
                    statistic_variance = residual[:, 5:].var(axis=(1, 2))  # per-direction window variance
                    D2 = (D_field**2).sum(1).cpu().numpy()
                    contraction = 1 - (2 * rate - rate * rate) * k / P
                    predicted_stationary = rate / (2 - rate) * P * statistic_variance
                    j = np.arange(1, squared_error.shape[1] + 1)
                    predicted = (
                        contraction ** j[None, :] * D2[:, None]
                        + (1 - contraction ** j[None, :]) * predicted_stationary[:, None]
                    )
                    tail = slice(int(0.7 * squared_error.shape[1]), None)
                    rows.append(
                        dict(
                            num_units=n,
                            num_parameters=P,
                            num_tangents=k,
                            tracking_rate=rate,
                            window=window,
                            bias_field_sq_norm=D2.tolist(),
                            stationary_error_measured=squared_error[:, tail].mean(1).tolist(),
                            stationary_error_predicted=predicted_stationary.tolist(),
                            predicted_contraction=float(contraction),
                            error_curve_measured=squared_error.mean(0).tolist(),
                            error_curve_predicted=predicted.mean(0).tolist(),
                        )
                    )
                    print(
                        f"   k={k:2d} beta={rate:4.2f} w={window:3d}: stationary |b-D|^2 meas/pred = "
                        f"{np.median(squared_error[:, tail].mean(1)):.3e}/{np.median(predicted_stationary):.3e} "
                        f"(|D|^2={np.median(D2):.2e})  curve ratio meas/pred at j=(10,50,200): "
                        f"{np.median(squared_error[:, 9] / predicted[:, 9]):.2f},"
                        f"{np.median(squared_error[:, 49] / predicted[:, 49]):.2f},"
                        f"{np.median(squared_error[:, 199] / predicted[:, 199]):.2f}",
                        flush=True,
                    )
        record = dict(args=stored_arguments(args), num_units=n, num_parameters=P, configurations=rows)
        write_json(args.output_dir / f"n{n}.json", record)


if __name__ == "__main__":
    main()
