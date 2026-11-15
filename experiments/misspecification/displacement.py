"""Displacement of the limit under misspecification.

The teacher has more units than the student; the modelled units are driven by the unmodelled ones with
strength ``epsilon`` (``epsilon = 0`` is well specified).  Per system, lanes (RTRL, SnAp-1, immediate,
leak) share data, start from the student block of the teacher and use ``Gamma = G^-1``.  The RTRL lane
gives ``theta_0(epsilon)``; at ``theta_0`` the script measures ``v_Q = V_Q(theta_0)`` and the full Jacobian
``DV_Q(theta_0)`` (central differences, common random numbers) and compares

    measured displacement  theta_Q - theta_0     with     -DV_Q^-1 v_Q,

and the risk gap ``L(theta_Q) - L(theta_0)`` with its quadratic prediction.  Theory: the displacement is
of first order in ``epsilon`` and the gap of second order.

    python -m experiments.misspecification.displacement --couplings 0.05,0.1,0.2,0.4
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments.common import float_list, make_parser, stored_arguments, write_json
from trace_instruments import DTYPE, rules
from trace_instruments.learning import evaluate_risk, run_online_learning
from trace_instruments.stationary_moments import estimate_stationary_moments
from trace_instruments.teacher_student import TeacherStudentSystem

RESULTS_SUBDIR = ("misspecification", "displacement")
RULES = [rules.RTRL, rules.SNAP1, rules.IMMEDIATE, "leak:0.5"]
PRECONDITIONER_MOMENT_STEPS = 30_000
LEARNING_SEED, JACOBIAN_SEED, EVALUATION_SEED = 777, 31337, 4242


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR)
    parser.add_argument("--teacher-units", type=int, default=10)
    parser.add_argument("--student-units", type=int, default=5)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--num-systems", type=int, default=8)
    parser.add_argument("--couplings", default="0.05,0.1,0.2,0.4", help="comma-separated values of epsilon")
    parser.add_argument("--learning-steps", type=int, default=300_000)
    parser.add_argument("--moment-steps", type=int, default=200_000)
    parser.add_argument("--evaluation-steps", type=int, default=200_000)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--gain", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finite-difference-step", type=float, default=2e-3)
    args = parser.parse_args(argv)
    num_rules = len(RULES)
    S, h = args.num_systems, args.finite_difference_step
    for coupling in float_list(args.couplings):
        start = time.time()

        def make_system(replicas):
            return TeacherStudentSystem(
                args.teacher_units,
                args.student_units,
                args.input_dim,
                S,
                args.device,
                replicas=replicas,
                family="random",
                gain=args.gain,
                noise_std=args.noise_std,
                seed=args.seed,
                coupling=coupling,
            )

        system = make_system(1)
        P = system.student.P
        G0 = estimate_stationary_moments(system, [], PRECONDITIONER_MOMENT_STEPS).cross_gram[rules.RTRL]
        Gamma = torch.linalg.inv(0.5 * (G0 + G0.transpose(1, 2)))
        system = make_system(num_rules)
        system.generator.manual_seed(LEARNING_SEED)
        out = run_online_learning(
            system,
            [RULES[i % num_rules] for i in range(system.B)],
            args.step_size,
            args.learning_steps,
            preconditioner=Gamma.repeat_interleave(num_rules, 0),
            record_covariance=False,
            average_from=0.5,
        )
        averaged = out["averaged_parameters"].view(S, num_rules, P)
        diverged = out["diverged"].view(S, num_rules).cpu().numpy()
        theta0 = averaged[:, 0].contiguous()

        # Full Jacobians at theta_0 by central differences with common random numbers: away from calibration
        # DV_Q = E[(dQ)^T s] + E[Q^T H Psi], and the Hessian of L is not the Gauss-Newton matrix.
        K = 2 * P + 1
        jacobian_system = make_system(K)
        jacobian_system.generator.manual_seed(JACOBIAN_SEED)
        theta_grid = theta0[:, None, :].repeat(1, K, 1)
        offsets = torch.eye(P, dtype=DTYPE, device=args.device) * h
        theta_grid[:, 1::2] += offsets
        theta_grid[:, 2::2] -= offsets
        moments = estimate_stationary_moments(
            jacobian_system, RULES[1:], args.moment_steps, theta=theta_grid.reshape(-1, P)
        )
        C, V = moments.cross_gram, moments.mean_update

        def at_theta0(x):
            return x.view(S, K, *x.shape[1:])[:, 0]

        def jacobian(x):
            """[b, p, j] = d V_p / d theta_j"""
            x = x.view(S, K, P)
            return ((x[:, 1::2] - x[:, 2::2]) / (2 * h)).transpose(1, 2).cpu().numpy()

        hessian = jacobian(V[rules.RTRL])
        hessian = 0.5 * (hessian + hessian.transpose(0, 2, 1))
        gradient0 = at_theta0(V[rules.RTRL]).cpu().numpy()
        evaluation = make_system(num_rules)
        evaluation.generator.manual_seed(EVALUATION_SEED)
        risk, _ = evaluate_risk(evaluation, averaged.reshape(-1, P), args.evaluation_steps)
        risk = risk.view(S, num_rules).cpu().numpy()
        # risk still to gain from theta_0
        newton = np.array([0.5 * gradient0[b] @ np.linalg.solve(hessian[b], gradient0[b]) for b in range(S)])
        gauss_newton = at_theta0(C[rules.RTRL]).cpu().numpy()
        record = dict(
            args=stored_arguments(args),
            coupling=coupling,
            num_parameters=P,
            rules={},
            reference_gradient_norm=np.linalg.norm(gradient0, axis=1).tolist(),
            reference_loss=risk[:, 0].tolist(),
            newton_decrement=newton.tolist(),
            hessian_min_eig=[float(np.linalg.eigvalsh(hessian[b]).min()) for b in range(S)],
            gauss_newton_vs_hessian=[
                float(np.linalg.norm(gauss_newton[b] - hessian[b]) / np.linalg.norm(hessian[b])) for b in range(S)
            ],
        )
        print(
            f"eps={coupling}: |grad L(theta_0)| median {np.median(record['reference_gradient_norm']):.2e}, "
            f"Newton decrement {np.median(newton):.2e}, loss {np.median(risk[:, 0]):.4e}, "
            f"|GN - Hessian|/|Hessian| {np.median(record['gauss_newton_vs_hessian']):.3f}  "
            f"[{time.time() - start:.0f}s]",
            flush=True,
        )

        def row_norm(x):
            return np.linalg.norm(x, axis=1)

        for r, rule in enumerate(RULES[1:], start=1):
            displacement = (averaged[:, r] - theta0).cpu().numpy()
            C_rule = at_theta0(C[rule]).cpu().numpy()
            DV = jacobian(V[rule])
            v = at_theta0(V[rule]).cpu().numpy()
            # the theorem: full Jacobian
            predicted = np.stack([-np.linalg.solve(DV[b], v[b]) for b in range(S)])
            # cross-Gram only, for comparison
            predicted_gram = np.stack([-np.linalg.solve(C_rule[b], v[b]) for b in range(S)])

            def cosine(x):
                return (displacement * x).sum(1) / (row_norm(displacement) * row_norm(x) + 1e-300)

            gap = risk[:, r] - risk[:, 0]
            predicted_gap = np.array([0.5 * predicted[b] @ hessian[b] @ predicted[b] for b in range(S)])
            ok = ~diverged[:, r]
            jacobian_vs_gram = [float(np.linalg.norm(DV[b] - C_rule[b]) / np.linalg.norm(DV[b])) for b in range(S)]
            record["rules"][rule] = dict(
                displacement=row_norm(displacement).tolist(),
                predicted_displacement=row_norm(predicted).tolist(),
                direction_cosine=cosine(predicted).tolist(),
                predicted_displacement_cross_gram=row_norm(predicted_gram).tolist(),
                direction_cosine_cross_gram=cosine(predicted_gram).tolist(),
                risk_gap=gap.tolist(),
                predicted_risk_gap=predicted_gap.tolist(),
                mean_update_norm=row_norm(v).tolist(),
                diverged=diverged[:, r].tolist(),
                jacobian_vs_cross_gram=jacobian_vs_gram,
            )
            print(
                f"   {rule:9s} |v|={np.median(row_norm(v)):.2e} "
                f"|disp| meas/pred={np.median(row_norm(displacement)[ok]):.3e}/{np.median(row_norm(predicted)[ok]):.3e} "
                f"ratio={np.median((row_norm(displacement) / row_norm(predicted))[ok]):.2f} "
                f"cos={np.median(cosine(predicted)[ok]):+.3f} "
                f"[cross-Gram only: ratio={np.median((row_norm(displacement) / row_norm(predicted_gram))[ok]):.2f} "
                f"cos={np.median(cosine(predicted_gram)[ok]):+.3f}; |DV-C|/|DV|={np.median(jacobian_vs_gram):.3f}] "
                f"gap meas/pred={np.median(gap[ok]):.3e}/{np.median(predicted_gap[ok]):.3e} "
                f"div={int(diverged[:, r].sum())}",
                flush=True,
            )
        write_json(args.output_dir / f"coupling_{coupling}.json", record)


if __name__ == "__main__":
    main()
