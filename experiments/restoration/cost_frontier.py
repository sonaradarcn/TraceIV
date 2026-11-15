"""What restoration costs: risk against arithmetic for restoration by tangent probes.

Well-specified tanh teacher-student with ``m`` of ``n`` units read out, so SnAp-1 is blind to the
``P_e = (n - m) q`` incoming weights of the unread units.  All lanes of a system share data, start and
step size.

    rule        description                                                  arithmetic per step
    rtrl        RTRL                                                         n^2 P
    snap1       SnAp-1 alone                                                 P
    hybrid:m    SnAp-1 on the read units, exact sensitivity for the unread   n^2 P_e + P
    probes      SnAp-1 + restoration from k tangents confined to the blind   k (n^2 + n q + P) + P
                rows, for a grid of k, tracking rate and window

Costs are multiply-adds per step of the recursions (no constant factors);
``experiments.restoration.step_cost_benchmark`` times them.  Output: paired excess risk of every lane.

    python -m experiments.restoration.cost_frontier --num-units 16 --num-readout 8,14
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import int_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import evaluate_risk, run_online_learning
from trace_instruments.restoration import PROJECTED, BiasFieldEstimator, LaneGroupedEstimators
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("restoration", "cost_frontier")
EVALUATION_SEED = 4242


def arithmetic_cost(rule, probes, n, q, P, num_blind):
    """Multiply-adds per step of the recursions of one lane."""
    if rule == rules.RTRL:
        base = n * n * P
    elif rules.is_hybrid(rule):
        base = n * n * num_blind + P
    else:
        base = P
    return base + (probes["num_tangents"] * (n * n + n * q + P) if probes else 0)


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--num-units", type=int, default=16)
    parser.add_argument("--num-readout", default="8,14", help="comma-separated numbers m of units read out")
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=8)
    parser.add_argument("--learning-steps", type=int, default=150_000)
    parser.add_argument("--evaluation-steps", type=int, default=30_000)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--num-tangents", default="1,4,16,full", help="'full' probes all blind coordinates")
    parser.add_argument("--tracking-rates", default="0.05,0.3,1.0")
    parser.add_argument("--windows", default="50,200")
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    n, S = args.num_units, args.num_systems
    for m in int_list(args.num_readout):
        start = time.time()
        q = n + args.input_dim  # fan-in of a student unit (no bias)
        P = n * q
        num_blind = (n - m) * q
        # lanes of one system: (rule, probe settings or None)
        lanes = [(rules.RTRL, None), (rules.SNAP1, None), (f"{rules.HYBRID_PREFIX}{m}", None)]
        for k in args.num_tangents.split(","):
            num_tangents = num_blind if k == "full" else int(k)
            if num_tangents > num_blind:
                continue
            for rate in args.tracking_rates.split(","):
                for window in args.windows.split(","):
                    lanes.append(
                        (rules.SNAP1, dict(num_tangents=num_tangents, tracking_rate=float(rate), window=int(window)))
                    )
        replicas = len(lanes)

        def make_system(replicas):
            return TeacherStudentSystem(
                n,
                n,
                args.input_dim,
                S,
                args.device,
                replicas=replicas,
                family="random",
                gain=args.gain,
                noise_std=args.noise_std,
                seed=args.seed,
                num_readout=m,
                readout="units",
            )

        system = make_system(replicas)
        B = system.B
        generator = torch.Generator().manual_seed(args.seed + 7)
        direction = torch.randn(S, P, generator=generator, dtype=DTYPE)
        direction = direction / direction.norm(dim=1, keepdim=True)
        theta_star = system.initial_parameters.view(S, replicas, P)[:, 0]
        theta0 = (
            theta_star + args.perturbation * theta_star.norm(dim=1, keepdim=True) * direction.to(args.device)
        ).repeat_interleave(replicas, 0)
        support = torch.zeros(n, q, dtype=torch.bool)
        support[m:] = True
        support = support.reshape(-1)
        groups = []
        for j, (_, probes) in enumerate(lanes):
            if probes is None:
                continue
            lane_index = torch.arange(j, B, replicas, device=args.device)
            estimator = BiasFieldEstimator(
                system.student,
                S,
                args.device,
                mode=PROJECTED,
                num_tangents=probes["num_tangents"],
                window=probes["window"],
                burn_in=min(20, probes["window"] // 3),
                rate=probes["tracking_rate"],
                seed=args.seed + j,
                support=support,
            )
            groups.append((estimator, lane_index))
        out = run_online_learning(
            system,
            [lanes[i % replicas][0] for i in range(B)],
            args.step_size,
            args.learning_steps,
            initial_parameters=theta0,
            restoration=LaneGroupedEstimators(groups, B, P, args.device),
            record_covariance=False,
            average_from=0.7,
        )
        evaluation = make_system(replicas + 1)
        evaluation.generator.manual_seed(EVALUATION_SEED)
        theta_eval = torch.cat(
            [out["averaged_parameters"].view(S, replicas, P), theta_star[:, None, :]], dim=1
        ).reshape(-1, P)
        risk, _ = evaluate_risk(evaluation, theta_eval, args.evaluation_steps)
        risk = risk.view(S, replicas + 1).cpu().numpy()
        diverged = out["diverged"].view(S, replicas).cpu().numpy()
        excess = np.where(diverged, np.nan, risk[:, :replicas] - risk[:, replicas:])
        cost = [arithmetic_cost(rule, probes, n, q, P, num_blind) for rule, probes in lanes]
        record = dict(
            args=stored_arguments(args),
            num_units=n,
            num_readout=m,
            num_parameters=P,
            num_blind_parameters=num_blind,
            fan_in=q,
            lanes=[dict(rule=rule, probes=probes) for rule, probes in lanes],
            arithmetic_cost=cost,
            excess_risk=excess.tolist(),
            diverged=diverged.tolist(),
            seconds=time.time() - start,
        )
        write_json(args.output_dir / f"n{n}_m{m}.json", record)
        print(f"n={n} m={m} P={P} Pe={num_blind}  [{time.time() - start:.0f}s]", flush=True)
        for j, (rule, probes) in enumerate(lanes):
            median = np.nanmedian(excess[:, j]) if (~diverged[:, j]).any() else float("nan")
            print(
                f"   {rule:10s} {str(probes or ''):62s} cost/RTRL={cost[j] / cost[0]:.4f} "
                f"excess median={median:.3e} div={int(diverged[:, j].sum())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
