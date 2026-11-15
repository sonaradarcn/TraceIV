"""Relevance: what a sparse read-out removes.  Well-specified teacher-student with ``m`` of ``n`` units read.

Per system, five lanes share data: RTRL, SnAp-1, the immediate rule, and the two trace rules with
restoration.  All lanes start from the same perturbed parameter and use plain constant-step updates with a
common step, so RTRL and the cheap rules differ only in their instrument; the restored lanes add an
estimate of the bias field.  The script reports the rank and coherence of each instrument at ``theta*``
and the paired excess risk ``L(theta_bar) - L(theta*)``.

    python -m experiments.relevance.readout_rank --readout units
    python -m experiments.relevance.readout_rank --readout dense
    python -m experiments.relevance.readout_rank --cell gru_reset_after      # also gru_reset_before
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import family_tag, int_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import evaluate_risk, run_online_learning
from trace_instruments.restoration import EMA, EXACT, PROJECTED, UORO, BiasFieldEstimator
from trace_instruments.stationary_moments import estimate_stationary_moments, spectral_summary
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("relevance", "readout_rank")
RESTORED = "+restoration"
# (rule, with restoration) of the five lanes of a system
LANES = [
    (rules.RTRL, False),
    (rules.SNAP1, False),
    (rules.IMMEDIATE, False),
    (rules.SNAP1, True),
    (rules.IMMEDIATE, True),
]
LANE_NAMES = [rule + (RESTORED if restored else "") for rule, restored in LANES]
TRACE_RULES = (rules.RTRL, rules.SNAP1, rules.IMMEDIATE)
EVALUATION_SEED = 4242


def output_name(args, m):
    name = f"{args.cell}_{family_tag(args.family)}_g{args.gain}_{args.readout}_m{m}"
    if args.restoration != EXACT:
        name += f"_{args.restoration}_k{args.num_tangents}"
    return name + ".json"


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=16)
    parser.add_argument("--num-readout", default="6,3,1", help="comma-separated numbers m of units read out")
    parser.add_argument("--readout", default="units", choices=("units", "dense"))
    parser.add_argument("--family", default="random")
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--moment-steps", type=int, default=30_000)
    parser.add_argument("--learning-steps", type=int, default=150_000)
    parser.add_argument("--evaluation-steps", type=int, default=30_000)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--perturbation", type=float, default=0.15, help="relative distance of the start from theta*")
    parser.add_argument("--restoration", default=EXACT, choices=(EXACT, PROJECTED, EMA, UORO))
    parser.add_argument("--num-tangents", type=int, default=4, help="tangents of the projected / ema estimator")
    parser.add_argument("--tracking-rate", type=float, default=0.002)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell", default="tanh", help="tanh, gru_reset_after or gru_reset_before")
    args = parser.parse_args(argv)
    S = args.num_systems
    replicas = len(LANES)
    for m in int_list(args.num_readout):
        start = time.time()

        def make_system(replicas):
            return TeacherStudentSystem(
                args.num_units,
                args.num_units,
                args.input_dim,
                S,
                args.device,
                replicas=replicas,
                family=args.family,
                gain=args.gain,
                noise_std=args.noise_std,
                seed=args.seed,
                cell=args.cell,
                num_readout=m,
                readout=args.readout,
            )

        moment_system = make_system(1)
        moments = estimate_stationary_moments(moment_system, [rules.SNAP1, rules.IMMEDIATE], args.moment_steps)
        C, Sigma, cosine = moments.cross_gram, moments.instrument_gram, moments.gradient_cosine
        G = C[rules.RTRL]
        P = moment_system.student.P
        summaries = {rule: spectral_summary(C[rule], Sigma[rule], G) for rule in TRACE_RULES}
        system = make_system(replicas)
        generator = torch.Generator().manual_seed(args.seed + 7)
        direction = torch.randn(S, P, generator=generator, dtype=DTYPE)
        direction = direction / direction.norm(dim=1, keepdim=True)
        theta_star = moment_system.initial_parameters
        theta0 = (
            theta_star + args.perturbation * theta_star.norm(dim=1, keepdim=True) * direction.to(args.device)
        ).repeat_interleave(replicas, 0)
        lane_rules = [LANES[i % replicas][0] for i in range(system.B)]
        mask = torch.tensor(
            [1.0 if LANES[i % replicas][1] else 0.0 for i in range(system.B)], dtype=DTYPE, device=args.device
        )
        if args.restoration == EXACT:
            estimator = BiasFieldEstimator(system.student, system.B, args.device, mode=EXACT, rate=args.tracking_rate)
        else:
            estimator = BiasFieldEstimator(
                system.student,
                system.B,
                args.device,
                mode=args.restoration,
                num_tangents=args.num_tangents,
                window=args.window,
                rate=args.tracking_rate,
                seed=args.seed,
            )
        out = run_online_learning(
            system,
            lane_rules,
            args.step_size,
            args.learning_steps,
            initial_parameters=theta0,
            restoration=estimator,
            restoration_mask=mask,
            record_covariance=False,
            average_from=0.7,
        )
        # paired evaluation: same evaluation stream for every lane and for theta*
        evaluation = make_system(replicas + 2)
        evaluation.generator.manual_seed(EVALUATION_SEED)
        theta_eval = torch.cat(
            [
                out["averaged_parameters"].view(S, replicas, P),
                theta_star[:, None, :],
                theta0.view(S, replicas, P)[:, :1],
            ],
            dim=1,
        ).reshape(-1, P)
        risk, gradient = evaluate_risk(evaluation, theta_eval, args.evaluation_steps)
        risk = risk.view(S, replicas + 2).cpu().numpy()
        gradient_norm = gradient.norm(dim=1).view(S, replicas + 2).cpu().numpy()
        excess = risk[:, :replicas] - risk[:, replicas : replicas + 1]
        excess_start = risk[:, replicas + 1] - risk[:, replicas]
        diverged = out["diverged"].view(S, replicas).cpu().numpy()
        excess = np.where(diverged, np.nan, excess)  # diverged lanes are reset to theta*: not a result
        record = dict(
            args=stored_arguments(args),
            num_readout=m,
            num_parameters=P,
            lanes=LANE_NAMES,
            excess_risk=excess.tolist(),
            excess_risk_start=excess_start.tolist(),
            gradient_norm=gradient_norm[:, :replicas].tolist(),
            diverged=diverged.tolist(),
            seconds=time.time() - start,
            rank={rule: [s["rank"] for s in summaries[rule]] for rule in summaries},
            coherence={rule: [s["coherence"].tolist() for s in summaries[rule]] for rule in summaries},
            sym_min_eig={rule: [s["sym_min_eig"] for s in summaries[rule]] for rule in summaries},
            gradient_cosine={rule: cosine[rule].cpu().tolist() for rule in summaries},
        )
        write_json(args.output_dir / output_name(args, m), record)
        print(
            f"m={m} ({args.readout}) P={P} start excess (median) {np.median(excess_start):.3e}   "
            f"[{time.time() - start:.0f}s]"
        )
        for i, name in enumerate(LANE_NAMES):
            rule = name.split("+")[0]
            print(
                f"   {name:21s} rank={np.median([s['rank'] for s in summaries[rule]]):.0f}/{P} "
                f"rho2min={np.median([s['coherence'].min() for s in summaries[rule]]):.3f} "
                f"cos={np.median(cosine[rule].cpu().numpy()):.2f} "
                f"excess median={np.nanmedian(excess[:, i]):.3e} "
                f"[min {np.nanmin(excess[:, i]):.2e}, max {np.nanmax(excess[:, i]):.2e}] "
                f"|gradL|={np.median(gradient_norm[:, i]):.2e} div={int(diverged[:, i].sum())}",
                flush=True,
            )


if __name__ == "__main__":
    main()
