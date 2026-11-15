"""A drifting teacher: the lag of a constant-step rule behind a moving calibrated point.

The teacher's parameters move at a constant velocity, ``theta*_t = theta*_0 + t * drift``, so the stream
is not stationary and the calibrated point of the student moves with it.  For the mean iterate linearised
about the moving point, ``e_t = theta_t - theta*_t`` obeys

    e_{t+1} = (I - step_size * C_Q) e_t - drift,          e_0 = 0,

whose fixed point, when ``C_Q`` is positive stable, is the lag ``-(step_size * C_Q)^{-1} drift``; the drift
costs the excess risk ``0.5 e^T G e``.  In directions where ``C_Q`` is singular (the blind coordinates of
a unit-local trace under a sparse read-out) the recursion gives ``e_t = -t * drift``: the rule does not
follow at all.

Lanes.  Per system: every rule at every step size, once with the drifting teacher and once with the
teacher at rest; all lanes of a system share inputs and noise.  The measured lag is the difference between
the two errors ``theta_t - theta*_t``, which removes most of the fluctuation, averaged over the last fifth
of the run; the measured cost is the difference of the mean losses over the same window.  The prediction
iterates the recursion above with ``C_Q`` and ``G`` estimated at ``theta*_0``.

    python -m experiments.nonstationarity.tracking_lag
    python -m experiments.nonstationarity.tracking_lag --num-readout 3     # sparse read-out: blind coordinates
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import family_tag, float_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.config import DIVERGENCE_BOUND
from trace_instruments.stationary_moments import estimate_stationary_moments
from trace_instruments.teacher_student import TeacherStudentSystem
from trace_instruments.traces import lane_coefficients

RESULTS_SUBDIR = ("nonstationarity", "tracking_lag")
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5"]
WINDOW = 0.2  # the last fifth of the run


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


@torch.no_grad()
def run(system, lane_rules, step_size, drift, C, G, num_steps):
    """Advance all lanes; returns window averages of the error, of its prediction and of the loss.

    ``drift`` (B, P) is the teacher's velocity per lane (zero for the lanes at rest); ``C`` (B, P, P) the
    cross-Gram of the lane's rule and ``G`` (B, P, P) the Gauss-Newton matrix, both at the initial point.
    """
    student, device = system.student, system.device
    B, P = system.B, student.P
    exact_coef, snap_coef, leak_coef = lane_coefficients(lane_rules, device)
    theta = system.initial_parameters.clone()
    Z = torch.zeros(B, student.n, P, dtype=DTYPE, device=device)
    predicted = torch.zeros(B, P, dtype=DTYPE, device=device)
    alive = torch.ones(B, dtype=torch.bool, device=device)
    start = int((1 - WINDOW) * num_steps)
    sums = {key: torch.zeros(B, P, dtype=DTYPE, device=device) for key in ("error", "predicted_error")} | {
        key: torch.zeros(B, dtype=DTYPE, device=device) for key in ("loss", "predicted_cost")
    }
    system.reset_state()
    for t in range(num_steps):
        delta, D, phi, W, err = system.step(theta)
        A = D[:, :, None] * W
        propagation = exact_coef[:, None, None] * A + torch.diag_embed(
            snap_coef[:, None] * torch.diagonal(A, dim1=1, dim2=2) + leak_coef[:, None]
        )
        Z = torch.bmm(propagation, Z) + student.immediate_jacobian(D, phi)
        update = torch.einsum("bkp,bk->bp", Z, delta)
        theta = theta - step_size[:, None] * update * alive[:, None]
        system.teacher.theta += drift
        predicted = predicted - step_size[:, None] * torch.einsum("bpq,bq->bp", C, predicted) - drift
        bad = ~torch.isfinite(theta).all(1) | (theta.abs().amax(1) > DIVERGENCE_BOUND)
        if bad.any():
            alive &= ~bad
            theta = torch.where(bad[:, None], system.initial_parameters, theta)
            system.student_state = torch.where(
                bad[:, None], torch.zeros_like(system.student_state), system.student_state
            )
            Z[bad] = 0
        if t >= start:
            sums["error"] += theta - system.teacher.theta
            sums["predicted_error"] += predicted
            sums["loss"] += 0.5 * (err * err).sum(1)
            sums["predicted_cost"] += 0.5 * torch.einsum("bp,bpq,bq->b", predicted, G, predicted)
    count = num_steps - start
    out = {key: value / count for key, value in sums.items()}
    out["final_error"] = theta - system.teacher.theta
    out["final_predicted_error"] = predicted
    out["diverged"] = ~alive
    return out


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--cell", default="tanh")
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=8)
    parser.add_argument("--num-readout", type=int, default=None, help="units read out (all by default)")
    parser.add_argument("--family", default="random")
    parser.add_argument("--gain", type=float, default=0.9)
    parser.add_argument("--moment-steps", type=int, default=100_000, help="steps of the moment estimate")
    parser.add_argument("--learning-steps", type=int, default=200_000)
    parser.add_argument("--step-sizes", default="0.01,0.03")
    parser.add_argument(
        "--total-drift", type=float, default=0.05, help="distance the teacher moves over the run, relative to |theta*|"
    )
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    start_time = time.time()
    step_sizes = float_list(args.step_sizes)
    S, num_rules, num_step_sizes = args.num_systems, len(RULES), len(step_sizes)
    replicas = num_rules * num_step_sizes * 2  # lane = (rule, step size, drifting or at rest)
    shape = (S, num_rules, num_step_sizes, 2)

    def make_system(lanes_per_system):
        return TeacherStudentSystem(
            args.num_units,
            args.num_units,
            args.input_dim,
            S,
            args.device,
            replicas=lanes_per_system,
            family=args.family,
            gain=args.gain,
            num_readout=args.num_readout,
            noise_std=args.noise_std,
            seed=args.seed,
            cell=args.cell,
        )

    moments = estimate_stationary_moments(make_system(1), RULES, args.moment_steps)
    C = torch.stack([moments.cross_gram[rule] for rule in RULES], 1)  # (S, rules, P, P)
    G = moments.cross_gram[rules.RTRL]
    G = (G + G.transpose(1, 2)) / 2
    system = make_system(replicas)
    system.generator.manual_seed(777 + args.seed)
    P = system.student.P
    theta_star = system.initial_parameters.view(S, replicas, P)[:, 0]

    direction_generator = torch.Generator().manual_seed(4242 + args.seed)
    direction = torch.randn(S, P, generator=direction_generator, dtype=DTYPE).to(args.device)
    direction /= direction.norm(dim=1, keepdim=True)
    velocity = args.total_drift * theta_star.norm(dim=1, keepdim=True) / args.learning_steps * direction  # (S, P)

    def lanes(per_system_rule_step_flag):
        """Expand a (S, rules, step sizes, 2, ...) tensor to the lane order of the system."""
        return per_system_rule_step_flag.reshape(S * replicas, *per_system_rule_step_flag.shape[4:])

    ones = torch.ones(shape, dtype=DTYPE, device=args.device)
    lane_rules = [RULES[(lane % replicas) // (num_step_sizes * 2)] for lane in range(S * replicas)]
    lane_step = lanes(ones * torch.tensor(step_sizes, dtype=DTYPE, device=args.device)[None, None, :, None])
    drifting = torch.tensor([1.0, 0.0], dtype=DTYPE, device=args.device)[None, None, None, :]
    lane_drift = lanes((ones * drifting)[..., None] * velocity[:, None, None, None, :])
    lane_C = lanes(C[:, :, None, None].expand(S, num_rules, num_step_sizes, 2, P, P))
    lane_G = lanes(G[:, None, None, None].expand(S, num_rules, num_step_sizes, 2, P, P))

    out = run(system, lane_rules, lane_step, lane_drift, lane_C, lane_G, args.learning_steps)
    view = {key: value.view(*shape, *value.shape[1:]).cpu().numpy() for key, value in out.items()}
    C_np, velocity_np = C.cpu().numpy(), velocity.cpu().numpy()
    scale = theta_star.norm(dim=1).cpu().numpy()

    record = dict(
        args=stored_arguments(args),
        rules=RULES,
        step_sizes=step_sizes,
        num_parameters=P,
        drift_per_step=[float(np.linalg.norm(v)) for v in velocity_np],
        parameter_norm=scale.tolist(),
        results={},
    )
    for r, rule in enumerate(RULES):
        rank = [int(np.linalg.matrix_rank(C_np[s, r], tol=1e-8 * np.linalg.norm(C_np[s, r], 2))) for s in range(S)]
        min_real = [float(np.linalg.eigvals(C_np[s, r]).real.min()) for s in range(S)]
        record["results"][rule] = dict(rank=rank, min_real_eig=min_real, step_sizes={})
        for a, step_size in enumerate(step_sizes):
            rows = dict(
                lag_cosine=[], lag_norm_ratio=[], measured_lag=[], predicted_lag=[], steady_state_lag=[],
                final_lag_cosine=[], final_lag_norm_ratio=[], measured_cost=[], predicted_cost=[], diverged=[],
                spectral_radius=[],
            )  # fmt: skip
            for s in range(S):
                diverged = bool(view["diverged"][s, r, a].any())
                measured = view["error"][s, r, a, 0] - view["error"][s, r, a, 1]
                predicted = view["predicted_error"][s, r, a, 0]
                final = view["final_error"][s, r, a, 0] - view["final_error"][s, r, a, 1]
                final_predicted = view["final_predicted_error"][s, r, a, 0]
                steady = float("nan")
                if rank[s] == P and min_real[s] > 0:
                    steady = float(np.linalg.norm(np.linalg.solve(step_size * C_np[s, r], velocity_np[s])) / scale[s])
                nan = float("nan")
                rows["diverged"].append(diverged)
                # of the linearised mean recursion; one or more means it has no stable fixed point
                rows["spectral_radius"].append(
                    float(np.abs(np.linalg.eigvals(np.eye(P) - step_size * C_np[s, r])).max())
                )
                rows["lag_cosine"].append(nan if diverged else cosine(measured, predicted))
                rows["lag_norm_ratio"].append(
                    nan if diverged else float(np.linalg.norm(measured) / np.linalg.norm(predicted))
                )
                rows["measured_lag"].append(nan if diverged else float(np.linalg.norm(measured) / scale[s]))
                rows["predicted_lag"].append(float(np.linalg.norm(predicted) / scale[s]))
                rows["steady_state_lag"].append(steady)
                rows["final_lag_cosine"].append(nan if diverged else cosine(final, final_predicted))
                rows["final_lag_norm_ratio"].append(
                    nan if diverged else float(np.linalg.norm(final) / np.linalg.norm(final_predicted))
                )
                rows["measured_cost"].append(
                    nan if diverged else float(view["loss"][s, r, a, 0] - view["loss"][s, r, a, 1])
                )
                rows["predicted_cost"].append(float(view["predicted_cost"][s, r, a, 0]))
            record["results"][rule]["step_sizes"][str(step_size)] = rows
            print(
                f"{rule:9s} alpha={step_size:<6g} rank {int(np.median(rank))}/{P} "
                f"lag meas/pred (|e|/|theta*|) {np.nanmedian(rows['measured_lag']):.3e}/{np.median(rows['predicted_lag']):.3e} "
                f"cos {np.nanmedian(rows['lag_cosine']):.3f} norm ratio {np.nanmedian(rows['lag_norm_ratio']):.3f} "
                f"cost meas/pred {np.nanmedian(rows['measured_cost']):.3e}/{np.median(rows['predicted_cost']):.3e} "
                f"div {sum(rows['diverged'])}",
                flush=True,
            )
    record["seconds"] = time.time() - start_time
    readout = "all" if args.num_readout is None else str(args.num_readout)
    name = f"{args.cell}_n{args.num_units}_{family_tag(args.family)}_g{args.gain}_m{readout}.json"
    write_json(args.output_dir / name, record)


if __name__ == "__main__":
    main()
