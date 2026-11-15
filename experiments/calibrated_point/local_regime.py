"""How far from a calibrated point the local picture holds.

Well-specified teacher-student (tanh or gated).  Every rule is started at ``theta* + r |theta*| u`` for a
common random unit direction ``u`` and a range of radii ``r``, and at an independent random draw of the
same family (``random_draw``), on the same data, with plain constant-step updates.  The script reports the
paired excess risk of the averaged iterate.

    python -m experiments.calibrated_point.local_regime --cell tanh
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import float_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import evaluate_risk, run_online_learning
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("calibrated_point", "local_regime")
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5"]
RANDOM_DRAW = "random_draw"
EVALUATION_SEED = 4242


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--cell", default="tanh")
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=16)
    parser.add_argument("--num-readout", type=int, default=0, help="units read out; 0 reads all of them")
    parser.add_argument("--readout", default="units", choices=("units", "dense"))
    parser.add_argument("--radii", default="0.05,0.1,0.2,0.4,0.8,1.6")
    parser.add_argument("--learning-steps", type=int, default=300_000)
    parser.add_argument("--evaluation-steps", type=int, default=30_000)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    start = time.time()
    radii = float_list(args.radii)
    num_starts = len(radii) + 1
    num_rules = len(RULES)
    replicas = num_starts * num_rules
    m = args.num_readout or args.num_units
    S = args.num_systems

    def make_system(replicas, seed=args.seed):
        return TeacherStudentSystem(
            args.num_units,
            args.num_units,
            args.input_dim,
            S,
            args.device,
            replicas=replicas,
            family="random",
            gain=args.gain,
            noise_std=args.noise_std,
            seed=seed,
            cell=args.cell,
            num_readout=m,
            readout=args.readout,
        )

    system = make_system(replicas)
    P = system.student.P
    theta_star = system.initial_parameters.view(S, replicas, P)[:, 0]
    other_draw = make_system(1, args.seed + 1000).initial_parameters  # independent draw of the same family
    generator = torch.Generator().manual_seed(args.seed + 7)
    direction = torch.randn(S, P, generator=generator, dtype=DTYPE)
    direction = (direction / direction.norm(dim=1, keepdim=True)).to(args.device)
    starts = [theta_star + r * theta_star.norm(dim=1, keepdim=True) * direction for r in radii] + [other_draw]
    theta0 = torch.stack(starts, 1)[:, :, None, :].expand(S, num_starts, num_rules, P).reshape(-1, P).clone()
    out = run_online_learning(
        system,
        [RULES[i % num_rules] for i in range(system.B)],
        args.step_size,
        args.learning_steps,
        initial_parameters=theta0,
        record_covariance=False,
        average_from=0.8,
    )
    # paired evaluation: one evaluation stream for every lane and for theta*
    evaluation = make_system(replicas + 1)
    evaluation.generator.manual_seed(EVALUATION_SEED)
    risk, _ = evaluate_risk(
        evaluation,
        torch.cat([out["averaged_parameters"].view(S, replicas, P), theta_star[:, None, :]], 1).reshape(-1, P),
        args.evaluation_steps,
    )
    risk = risk.view(S, replicas + 1).cpu().numpy()
    diverged = out["diverged"].view(S, num_starts, num_rules).cpu().numpy()
    excess = np.where(diverged, np.nan, (risk[:, :replicas] - risk[:, replicas:]).reshape(S, num_starts, num_rules))
    evaluation = make_system(num_starts + 1)
    evaluation.generator.manual_seed(EVALUATION_SEED)
    start_risk, _ = evaluate_risk(
        evaluation,
        torch.cat([torch.stack(starts, 1), theta_star[:, None, :]], 1).reshape(-1, P),
        args.evaluation_steps,
    )
    start_risk = start_risk.view(S, num_starts + 1).cpu().numpy()
    excess_start = start_risk[:, :num_starts] - start_risk[:, num_starts:]
    distance = (out["averaged_parameters"].view(S, num_starts, num_rules, P) - theta_star[:, None, None, :]).norm(
        dim=-1
    ).cpu().numpy() / theta_star.norm(dim=1).cpu().numpy()[:, None, None]
    labels = radii + [RANDOM_DRAW]
    record = dict(
        args=stored_arguments(args),
        rules=RULES,
        radii=labels,
        num_parameters=P,
        excess_risk=excess.tolist(),
        excess_risk_start=excess_start.tolist(),
        relative_distance=distance.tolist(),
        diverged=diverged.tolist(),
        seconds=time.time() - start,
    )
    write_json(args.output_dir / f"{args.cell}_n{args.num_units}_{args.readout}_m{m}.json", record)
    print(
        f"{args.cell} n={args.num_units} P={P} readout={args.readout} m={m}  [{time.time() - start:.0f}s]", flush=True
    )
    reference = np.nanmedian(excess[:, 0, 0])
    for i, label in enumerate(labels):
        print(
            f" r={label!s:11s} start={np.median(excess_start[:, i]):.2e} "
            + "  ".join(
                f"{rule}: {np.nanmedian(excess[:, i, j]):.2e} "
                f"ok={int(np.nansum(excess[:, i, j] < 5 * reference))}/{S} div={int(diverged[:, i, j].sum())}"
                for j, rule in enumerate(RULES)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
