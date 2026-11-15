"""Direct test of the efficiency theorem with Polyak-Ruppert averaging.

Per system, ``M`` independent replicas of every rule start at ``theta*`` and run
``theta <- theta - step_size * G^-1 Z^T delta``.  The average of each replica over the last
``(1 - burn_in_fraction) T`` steps has, by the central limit theorem,

    T_avg * E[(theta_bar - theta*)^T G (theta_bar - theta*)] / sigma^2  ->  tr(G C^-1 K C^-T) = sum_i rho_i^-2,

and ``E[theta_bar] = theta*``.  The script reports the measured constant against ``sum_i rho_i^-2`` from
the stationary moments, and a z-statistic for the mean displacement.

    python -m experiments.calibrated_point.efficiency_constant
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import family_tag, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import run_online_learning
from trace_instruments.stationary_moments import estimate_stationary_moments, spectral_summary
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("calibrated_point", "efficiency_constant")
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5", "leak:0.9"]


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=4)
    parser.add_argument("--num-replicas", type=int, default=32, help="independent runs per system and rule")
    parser.add_argument("--moment-steps", type=int, default=200_000)
    parser.add_argument("--learning-steps", type=int, default=100_000)
    parser.add_argument("--burn-in-fraction", type=float, default=0.2, help="part of the run left out of the average")
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--families", default="normal@0.9,random@0.9,chain@1.0,schur:1.0@0.9")
    parser.add_argument("--cell", default="tanh")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    num_rules = len(RULES)
    S, M = args.num_systems, args.num_replicas
    for item in args.families.split(","):
        family, gain = item.split("@")
        gain = float(gain)
        start = time.time()

        def make_system(independent_streams, replicas):
            return TeacherStudentSystem(
                args.num_units,
                args.num_units,
                args.input_dim,
                S,
                args.device,
                replicas=replicas,
                independent_streams=independent_streams,
                family=family,
                gain=gain,
                noise_std=args.noise_std,
                seed=args.seed,
                cell=args.cell,
            )

        system = make_system(1, 1)
        P = system.student.P
        moments = estimate_stationary_moments(system, RULES, args.moment_steps)
        C, Sigma, cosine = moments.cross_gram, moments.instrument_gram, moments.gradient_cosine
        G = C[rules.RTRL].cpu().numpy()
        G = (G + G.transpose(0, 2, 1)) / 2
        Gamma = torch.tensor(np.linalg.inv(G), dtype=DTYPE, device=args.device)
        learner_system = make_system(M, num_rules)
        learner_system.generator.manual_seed(555 + args.seed)
        out = run_online_learning(
            learner_system,
            [RULES[i % num_rules] for i in range(learner_system.B)],
            args.step_size,
            args.learning_steps,
            preconditioner=Gamma.repeat_interleave(M * num_rules, 0),
            record_covariance=False,
            average_from=args.burn_in_fraction,
        )
        averaged = out["averaged_parameters"].view(S, M, num_rules, P).cpu().numpy()
        diverged = out["diverged"].view(S, M, num_rules).cpu().numpy()
        theta_star = system.initial_parameters.cpu().numpy()
        averaged_steps = args.learning_steps * (1 - args.burn_in_fraction)
        record = dict(args=stored_arguments(args), family=family, gain=gain, num_parameters=P, rules={})
        for r, rule in enumerate(RULES):
            summaries = spectral_summary(C[rule], Sigma[rule], C[rules.RTRL])
            predicted = np.array([float(np.sum(1 / s["coherence"])) for s in summaries])
            measured, z = [], []
            for b in range(S):
                ok = ~diverged[b, :, r]
                d = averaged[b, ok, r] - theta_star[b]
                q = np.einsum("mp,pq,mq->m", d, G[b], d)
                measured.append(float(averaged_steps * q.mean() / args.noise_std**2))
                mu = d.mean(0)
                z.append(
                    float(
                        np.sqrt(
                            ok.sum() * averaged_steps * (mu @ G[b] @ mu) / args.noise_std**2 / max(predicted[b], 1e-300)
                        )
                    )
                )
            record["rules"][rule] = dict(
                predicted_constant=predicted.tolist(),
                measured_constant=measured,
                mean_displacement_z=z,
                gradient_cosine=cosine[rule].cpu().tolist(),
                diverged=int(diverged[:, :, r].sum()),
                coherence=[s["coherence"].tolist() for s in summaries],
            )
            print(
                f"{family}@{gain} {rule:9s} cos={np.median(cosine[rule].cpu().numpy()):.2f} "
                f"sum rho^-2 /P: pred={np.median(predicted) / P:6.3f} meas={np.median(measured) / P:6.3f} "
                f"ratio={np.median(np.array(measured) / predicted):.3f}  mean-displacement z={np.median(z):.2f} "
                f"(1 expected under no bias) div={int(diverged[:, :, r].sum())}",
                flush=True,
            )
        record["seconds"] = time.time() - start
        write_json(args.output_dir / f"{args.cell}_{family_tag(family)}_g{gain}.json", record)


if __name__ == "__main__":
    main()
