"""Calibrated teacher-student systems with full read-out: validity, stability and stationary risk.

For every (family, rule) the stationary moments at ``theta*`` give the predictions: spectrum of
``Gamma C``, ``sym(C)``, coherence spectrum, efficiency loss, Lyapunov covariance and excess risk.  The
rule is then run from ``theta*`` with the common preconditioner ``Gamma`` (``G^-1`` or the identity) and
the bias of the averaged iterate, the stationary covariance and the excess risk are measured.
One lane per (system, rule); the lanes of a system share inputs and noise.

    python -m experiments.calibrated_point.stationary_risk                       # tanh cell, Gamma = G^-1
    python -m experiments.calibrated_point.stationary_risk --cell gru_reset_after \
        --preconditioner identity --step-size 0.02 --families random@0.9,normal@0.9,orthogonal@0.9,random@1.3
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import family_tag, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import run_online_learning
from trace_instruments.stationary_moments import (
    estimate_stationary_moments,
    lyapunov_covariance,
    spectral_summary,
)
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("calibrated_point", "stationary_risk")
FAMILIES = [
    ("normal", 0.9),
    ("orthogonal", 0.9),
    ("random", 0.9),
    ("random", 1.3),
    ("chain", 0.5),
    ("chain", 1.0),
    ("chain", 1.5),
    ("schur:1.0", 0.9),
    ("schur:3.0", 0.9),
]
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5", "leak:0.9"]
GAUSS_NEWTON = "gauss_newton"
IDENTITY = "identity"


def output_name(args, family, gain):
    suffix = "" if args.preconditioner == GAUSS_NEWTON else "_" + args.preconditioner
    return f"{args.cell}_n{args.num_units}_{family_tag(family)}_g{gain}{suffix}.json"


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--cell", default="tanh", help="tanh, gru_reset_after or gru_reset_before")
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=8)
    parser.add_argument("--moment-steps", type=int, default=200_000, help="steps of the moment estimate")
    parser.add_argument("--learning-steps", type=int, default=200_000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--families", default="", help="comma-separated family@gain; all nine when empty")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preconditioner", default=GAUSS_NEWTON, choices=(GAUSS_NEWTON, IDENTITY), help="Gamma: G^-1 or I"
    )
    args = parser.parse_args(argv)
    num_rules = len(RULES)
    S = args.num_systems
    selected = [f for f in FAMILIES if not args.families or f"{f[0]}@{f[1]}" in args.families.split(",")]
    for family, gain in selected:
        path = args.output_dir / output_name(args, family, gain)
        tag = path.stem
        if path.exists():
            continue
        start = time.time()

        def make_system(replicas):
            return TeacherStudentSystem(
                args.num_units,
                args.num_units,
                args.input_dim,
                S,
                args.device,
                replicas=replicas,
                family=family,
                gain=gain,
                noise_std=args.noise_std,
                seed=args.seed,
                cell=args.cell,
            )

        system = make_system(1)
        teacher_W = system.teacher.weight_matrices()[0]
        P = system.student.P
        moments = estimate_stationary_moments(system, RULES, args.moment_steps)
        C, Sigma, V, cosine = moments.cross_gram, moments.instrument_gram, moments.mean_update, moments.gradient_cosine
        G = C[rules.RTRL].cpu().numpy()
        G = (G + G.transpose(0, 2, 1)) / 2
        if args.preconditioner == GAUSS_NEWTON:
            Gamma = np.linalg.inv(G)
        else:
            Gamma = np.broadcast_to(np.eye(P), G.shape).copy()
        learner_system = make_system(num_rules)
        learner_system.generator.manual_seed(777 + args.seed)
        out = run_online_learning(
            learner_system,
            [RULES[i % num_rules] for i in range(learner_system.B)],
            args.step_size,
            args.learning_steps,
            preconditioner=torch.tensor(Gamma, dtype=DTYPE, device=args.device).repeat_interleave(num_rules, 0),
            average_from=0.25,
        )
        averaged = out["averaged_parameters"].view(S, num_rules, P).cpu().numpy()
        theta_star = system.initial_parameters.cpu().numpy()
        covariance = out["covariance"].view(S, num_rules, P, P).cpu().numpy()
        diverged = out["diverged"].view(S, num_rules).cpu().numpy()
        record = dict(
            args=stored_arguments(args),
            family=family,
            gain=gain,
            num_parameters=P,
            rules={},
            spectral_radius=[float(torch.linalg.eigvals(w).abs().max()) for w in teacher_W],
            spectral_norm=[float(torch.linalg.matrix_norm(w, 2)) for w in teacher_W],
        )
        for r, rule in enumerate(RULES):
            summaries = spectral_summary(C[rule], Sigma[rule], C[rules.RTRL])
            C_np, Sigma_np = C[rule].cpu().numpy(), Sigma[rule].cpu().numpy()
            predicted_excess, predicted_trace, preconditioned_min = [], [], []
            measured_excess, measured_trace, bias = [], [], []
            for b in range(S):
                eigenvalues = np.linalg.eigvals(Gamma[b] @ C_np[b])
                preconditioned_min.append(float(eigenvalues.real.min()))
                if eigenvalues.real.min() > 0:
                    X = lyapunov_covariance(C_np[b], Sigma_np[b], args.step_size, args.noise_std**2, Gamma[b])
                    predicted_trace.append(float(np.trace(X)))
                    predicted_excess.append(float(0.5 * np.trace(G[b] @ X)))
                else:
                    predicted_trace.append(float("nan"))
                    predicted_excess.append(float("nan"))
                displacement = averaged[b, r] - theta_star[b]
                if diverged[b, r]:
                    measured_excess.append(float("nan"))
                    measured_trace.append(float("nan"))
                    bias.append(float("nan"))
                else:
                    measured_excess.append(float(0.5 * np.trace(G[b] @ covariance[b, r])))
                    measured_trace.append(float(np.trace(covariance[b, r])))
                    bias.append(float(np.sqrt(displacement @ G[b] @ displacement)))
            record["rules"][rule] = dict(
                min_real_eig_preconditioned=preconditioned_min,
                sym_min_eig=[s["sym_min_eig"] for s in summaries],
                min_real_eig=[s["min_real_eig"] for s in summaries],
                efficiency_loss=[s["efficiency_loss"] for s in summaries],
                coherence=[s["coherence"].tolist() for s in summaries],
                gradient_cosine=cosine[rule].cpu().tolist(),
                predicted_excess_risk=predicted_excess,
                measured_excess_risk=measured_excess,
                predicted_trace_cov=predicted_trace,
                measured_trace_cov=measured_trace,
                bias_norm=bias,
                diverged=diverged[:, r].tolist(),
                mean_update_norm=[float(v.norm()) for v in V[rule]],
            )
            print(
                f"{tag} {rule:9s} minRe(GamC)={np.median(preconditioned_min):+.3f} "
                f"sym(C)min={np.median([s['sym_min_eig'] for s in summaries]):+.2e} "
                f"cos={np.median(cosine[rule].cpu().numpy()):.2f} "
                f"eff={np.median([s['efficiency_loss'] for s in summaries]):6.2f} "
                f"rho2min={np.median([s['coherence'].min() for s in summaries]):.3f} "
                f"excess meas/pred={np.nanmedian(measured_excess):.3e}/{np.nanmedian(predicted_excess):.3e} "
                f"ratio={np.nanmedian(np.array(measured_excess) / np.array(predicted_excess)):.3f} "
                f"biasG={np.nanmedian(bias):.1e} div={int(diverged[:, r].sum())}",
                flush=True,
            )
        record["seconds"] = time.time() - start
        write_json(path, record)


if __name__ == "__main__":
    main()
