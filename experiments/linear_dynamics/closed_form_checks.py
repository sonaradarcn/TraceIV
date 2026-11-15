"""Linear recurrent dynamics: checks of the closed forms and a sweep of non-normality (exact, no simulation).

Model ``h_t = A h_{t-1} + x_t``, white input, full read-out, ``theta = vec(A)``; the moments of every rule
are computed exactly by ``trace_instruments.linear_theory``.

Part 1 (``theorem_checks``) verifies the theorem on linear dynamics on random stable matrices: the closed
form of the cross-Gram of the immediate rule, its spectrum, and the numerical-range criterion for
``sym(C) > 0``.  Part 2 (``nonnormality_sweep``) sweeps non-normality at fixed spectrum,
``A(s) = Q (diag(lambda) + s N) Q^T`` with ``N`` strictly upper triangular.

    python -m experiments.linear_dynamics.closed_form_checks
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from experiments.common import make_parser, stored_arguments, write_json
from trace_instruments import rules
from trace_instruments.linear_theory import numerical_radius, summarise_rule

RESULTS_SUBDIR = ("linear_dynamics",)
OUTPUT_NAME = "closed_forms.json"
SWEEP_RULES = (rules.SNAP1, rules.IMMEDIATE, "leak:0.5")
UPPER_SCALES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
NUM_SPECTRA = 8


def check_theorem(A):
    """Compare the exact moments of the immediate rule on ``A`` with the statements of the theorem."""
    n = A.shape[0]
    summary, C, _, Pi = summarise_rule(A, rules.IMMEDIATE)
    closed_form = np.linalg.solve(np.eye(n * n) - np.kron(A, A), np.kron(np.eye(n), Pi))
    Pi_half = np.real(sqrtm(Pi))
    B = np.linalg.solve(Pi_half, A @ Pi_half)
    AB = np.kron(A, B)
    criterion = float(np.linalg.eigvalsh((AB + AB.T) / 2).max())
    lam = np.linalg.eigvals(A)
    predicted_spectrum = np.array([1 / (1 - x * y) for x in lam for y in lam])
    spectrum = np.linalg.eigvals(np.kron(np.eye(n), np.linalg.inv(Pi)) @ C)
    spectrum_error = float(max(np.abs(predicted_spectrum - value).min() for value in spectrum))
    return dict(
        closed_form_error=float(np.abs(C - closed_form).max() / np.abs(C).max()),
        sym_positive=bool(summary["sym_min_eig"] > 0),
        criterion=criterion,
        criterion_agrees=bool((criterion < 1) == (summary["sym_min_eig"] > 0)),
        spectrum_error=spectrum_error,
        numerical_radius=float(numerical_radius(A)),
        norm_B=float(np.linalg.norm(B, 2)),
        norm_B_predicted=float(np.sqrt(1 - 1 / np.linalg.eigvalsh(Pi).max())),
    )


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR, device=None)
    parser.add_argument("--num-units", type=int, default=6)
    parser.add_argument("--num-matrices", type=int, default=20, help="random stable matrices in part 1")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(0)
    n = args.num_units
    record = dict(args=stored_arguments(args), theorem_checks=[], nonnormality_sweep=[])

    # ---- Part 1: theorem checks on random stable matrices
    for _ in range(args.num_matrices):
        A = rng.standard_normal((n, n))
        A *= rng.uniform(0.5, 0.95) / np.abs(np.linalg.eigvals(A)).max()
        record["theorem_checks"].append(check_theorem(A))
    checks = record["theorem_checks"]
    print(
        f"Part 1 ({args.num_matrices} matrices): max rel formula err {max(c['closed_form_error'] for c in checks):.1e}; "
        f"criterion agrees {sum(c['criterion_agrees'] for c in checks)}/{len(checks)}; "
        f"max spectrum err {max(c['spectrum_error'] for c in checks):.1e}; "
        f"max | ||B|| - pred | {max(abs(c['norm_B'] - c['norm_B_predicted']) for c in checks):.1e}; "
        f"w(A)<=1 implies sym>0: {all(c['sym_positive'] for c in checks if c['numerical_radius'] <= 1)} "
        f"({sum(c['numerical_radius'] <= 1 for c in checks)} cases)"
    )

    # ---- Part 2: non-normality sweep at fixed spectrum
    for seed in range(NUM_SPECTRA):
        spectrum_rng = np.random.default_rng(100 + seed)
        Q, _ = np.linalg.qr(spectrum_rng.standard_normal((n, n)))
        lam = spectrum_rng.uniform(0.3, 0.8, n) * spectrum_rng.choice([-1, 1], n)
        N = np.triu(spectrum_rng.standard_normal((n, n)), 1) / np.sqrt(n)
        for upper_scale in UPPER_SCALES:
            A = Q @ (np.diag(lam) + upper_scale * N) @ Q.T
            row = dict(
                seed=seed,
                upper_scale=upper_scale,
                spectral_radius=float(np.abs(lam).max()),
                numerical_radius=float(numerical_radius(A)),
                spectral_norm=float(np.linalg.norm(A, 2)),
                rules={rule: summarise_rule(A, rule)[0] for rule in SWEEP_RULES},
            )
            record["nonnormality_sweep"].append(row)
    print(
        "Part 2 (median over 8 spectra)   s    w(A)   | immediate: sym_min  minRe(S^-1 C)  eff   rho2_min "
        "| snap1: sym_min  minRe  eff  rho2_min"
    )
    for upper_scale in UPPER_SCALES:
        rows = [row for row in record["nonnormality_sweep"] if row["upper_scale"] == upper_scale]

        def median(value_of):
            return np.nanmedian([value_of(row) for row in rows])

        immediate, snap1 = (lambda row: row["rules"][rules.IMMEDIATE]), (lambda row: row["rules"][rules.SNAP1])
        print(
            f"   {upper_scale:4.2f}  {median(lambda row: row['numerical_radius']):6.3f} | "
            f"{median(lambda row: immediate(row)['sym_min_eig']):+9.3f} "
            f"{median(lambda row: immediate(row)['min_real_eig_whitened']):+8.3f} "
            f"{median(lambda row: immediate(row)['efficiency_loss']):9.2f} "
            f"{median(lambda row: immediate(row)['coherence_min']):8.4f} | "
            f"{median(lambda row: snap1(row)['sym_min_eig']):+9.3f} "
            f"{median(lambda row: snap1(row)['min_real_eig_whitened']):+8.3f} "
            f"{median(lambda row: snap1(row)['efficiency_loss']):9.2f} "
            f"{median(lambda row: snap1(row)['coherence_min']):8.4f}  "
            f"snap1 trace unstable in {sum(1 for row in rows if snap1(row).get('trace_unstable'))}/8, "
            f"snap1 minRe<0 in {sum(1 for row in rows if snap1(row)['min_real_eig_whitened'] < 0)}/8"
        )
    write_json(args.output_dir / OUTPUT_NAME, record)


if __name__ == "__main__":
    main()
