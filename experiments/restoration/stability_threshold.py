"""Restoration as damping: SnAp-1 leaves a calibrated point, restoration above a threshold holds it.

Linear model of the theorem on linear dynamics (``h_t = A h_{t-1} + x_t``, ``theta = vec A``) on matrices
where SnAp-1 is a valid instrument whose cross-Gram has an eigenvalue with negative real part.  Prediction,
from the exact moments ``C`` and ``G``:

* SnAp-1 alone drifts away from ``theta*`` at rate ``step_size * |min Re eig C|``;
* with ideal restoration the linearised discrete map
  ``[[I - alpha C, -alpha I], [beta (G - C), (1 - beta) I]]`` has spectral radius below one if and only if
  the tracking rate ``beta`` exceeds a threshold (computed here), and the lanes then stay at ``theta*``.

Simulation: lanes start at ``theta*`` on the same data with different tracking rates.  The script reports
the growth of ``|theta - theta*|``.

    python -m experiments.restoration.stability_threshold
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.linear_theory import exact_moments, restored_map_radius, schur_form_matrix

RESULTS_SUBDIR = ("restoration", "stability_threshold")
ESCAPE_RADIUS = 5.0
REFERENCE_LANES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE]


@torch.no_grad()
def simulate(A, step_size, tracking_rates, num_steps, noise_std, device, num_replicas=8, seed=0, num_records=200):
    """Distance from ``theta*`` over time, ``(num_replicas, num_lanes, num_records)``.

    Lanes per replica: RTRL, SnAp-1 alone, the immediate rule, then SnAp-1 with ideal restoration at each
    tracking rate.  The lanes of a replica share inputs and noise.
    """
    n = A.shape[0]
    P = n * n
    num_lanes = len(tracking_rates) + 3
    B = num_replicas * num_lanes
    generator = torch.Generator(device=device).manual_seed(seed)
    A_true = torch.tensor(A, dtype=DTYPE, device=device)
    theta_star = A_true.reshape(-1)
    theta = theta_star.expand(B, P).clone()
    rate = torch.tensor(([0.0, 0.0, 0.0] + list(tracking_rates)) * num_replicas, dtype=DTYPE, device=device)
    is_rtrl = torch.tensor(([1.0] + [0.0] * (num_lanes - 1)) * num_replicas, dtype=DTYPE, device=device)
    # 0 for the immediate rule (no trace memory)
    keeps_trace = torch.tensor(
        ([1.0, 1.0, 0.0] + [1.0] * len(tracking_rates)) * num_replicas, dtype=DTYPE, device=device
    )
    restored = (rate > 0).to(DTYPE)
    alive = torch.ones(B, dtype=DTYPE, device=device)
    teacher_state = torch.zeros(num_replicas, n, dtype=DTYPE, device=device).repeat_interleave(num_lanes, 0)
    student_state = teacher_state.clone()
    J = torch.zeros(B, n, P, dtype=DTYPE, device=device)
    Z = torch.zeros_like(J)
    b = torch.zeros(B, P, dtype=DTYPE, device=device)
    eye = torch.eye(n, dtype=DTYPE, device=device)
    distance = []
    for t in range(num_steps):
        x = torch.randn(num_replicas, n, generator=generator, dtype=DTYPE, device=device).repeat_interleave(
            num_lanes, 0
        )
        noise = noise_std * torch.randn(
            num_replicas, n, generator=generator, dtype=DTYPE, device=device
        ).repeat_interleave(num_lanes, 0)
        W = theta.view(B, n, n)
        # d(W h)/dW: the block of row i is h_{t-1}
        Bt = (eye[None, :, :, None] * student_state[:, None, None, :]).reshape(B, n, P)
        teacher_state = teacher_state @ A_true.T + x
        student_state = torch.einsum("bij,bj->bi", W, student_state) + x
        J = torch.bmm(W, J) + Bt
        Z = (keeps_trace[:, None] * torch.diagonal(W, dim1=1, dim2=2))[:, :, None] * Z + Bt
        delta = student_state - teacher_state - noise
        exact_update = torch.einsum("bnp,bn->bp", J, delta)
        trace_update = torch.einsum("bnp,bn->bp", Z, delta)
        # theta uses the current b ...
        g = is_rtrl[:, None] * exact_update + (1 - is_rtrl[:, None]) * (trace_update + restored[:, None] * b)
        theta = theta - step_size * torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) * alive[:, None]
        # ... then b is updated, as in the analysed map
        b = torch.nan_to_num(b + rate[:, None] * ((exact_update - trace_update) - b), nan=0.0, posinf=0.0, neginf=0.0)
        escaped = alive.bool() & (~torch.isfinite(theta).all(1) | ((theta - theta_star).norm(dim=1) > ESCAPE_RADIUS))
        if escaped.any():  # freeze escaped lanes at a sentinel distance equal to the escape radius
            alive = alive * (~escaped)
            theta = torch.where(escaped[:, None], (theta_star + ESCAPE_RADIUS / np.sqrt(P)).expand(B, P), theta)
            student_state = torch.where(escaped[:, None], teacher_state, student_state)
            J[escaped] = 0
            Z[escaped] = 0
            b[escaped] = 0
        if (t + 1) % (num_steps // num_records) == 0:
            distance.append((theta - theta_star).norm(dim=1).view(num_replicas, num_lanes).cpu())
    return torch.stack(distance, 2).numpy()


def threshold_rate(C, G, step_size):
    """Smallest tracking rate at which the restored linear map is stable (bisection on a log scale)."""
    low, high = 1e-8, 1.0
    for _ in range(60):
        mid = np.sqrt(low * high)
        if restored_map_radius(C, G, step_size, mid) < 1:
            high = mid
        else:
            low = mid
    return high


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR, device="cpu")
    parser.add_argument(
        "--systems", default="33:2,43:3,187:3,155:4", help="comma-separated seed:upper_scale of the Schur-form matrices"
    )
    parser.add_argument("--num-steps", type=int, default=400_000)
    parser.add_argument("--step-size", type=float, default=0.05, help="in units of 1 / lambda_max(G)")
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--rate-multiples", default="0.3,3,10,100", help="tracking rates as multiples of the threshold")
    parser.add_argument("--num-replicas", type=int, default=8)
    args = parser.parse_args(argv)
    multiples = args.rate_multiples.split(",")
    for item in args.systems.split(","):
        seed, upper_scale = item.split(":")
        seed, upper_scale = int(seed), float(upper_scale)
        A = schur_form_matrix(seed, upper_scale)
        C, _, _ = exact_moments(A, np.diag(np.diag(A)))
        G, _, _ = exact_moments(A, A)
        lambda_max = np.linalg.eigvalsh(G).max()
        step_size = args.step_size / lambda_max
        min_real = float(np.linalg.eigvals(C).real.min())
        threshold = threshold_rate(C, G, step_size)
        tracking_rates = [threshold * float(m) for m in multiples]
        start = time.time()
        distance = simulate(
            A,
            step_size,
            tracking_rates,
            args.num_steps,
            args.noise_std,
            args.device,
            num_replicas=args.num_replicas,
            seed=seed,
        )
        lanes = REFERENCE_LANES + [rules.restored_lane_name(m) for m in multiples]
        print(
            f"system {seed}:{upper_scale}  minRe eig(C)/lmax(G)={min_real / lambda_max:+.4f}  "
            f"predicted e-folding of SnAp-1 drift: {1 / (step_size * abs(min_real)):.0f} steps; "
            f"threshold beta={threshold:.2e} (beta/alpha={threshold / step_size / lambda_max:.4f} in units of lmax)  "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )
        for j, name in enumerate(lanes):
            median = np.median(distance[:, j, :], 0)
            num_escaped = int((distance[:, j, -1] > 1.0).sum())
            radius_note = ""
            if j >= 3:
                radius_note = (
                    f" predicted radius-1={restored_map_radius(C, G, step_size, tracking_rates[j - 3]) - 1:+.1e}"
                )
            print(
                f"   {name:36s} |theta-theta*| at 10%/50%/100% of run: "
                f"{median[19]:.3e} {median[99]:.3e} {median[-1]:.3e} "
                f"escaped {num_escaped}/{distance.shape[0]}{radius_note}",
                flush=True,
            )
        record = dict(
            args=stored_arguments(args),
            seed=seed,
            upper_scale=upper_scale,
            step_size=step_size,
            threshold_rate=threshold,
            tracking_rates=tracking_rates,
            min_real_eig=min_real,
            gauss_newton_max_eig=float(lambda_max),
            lanes=lanes,
            distance=distance.tolist(),
        )
        write_json(args.output_dir / f"system_{seed}_{upper_scale}.json", record)


if __name__ == "__main__":
    main()
