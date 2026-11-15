"""An exact example of a stable linear system in which the SnAp-1 cross-Gram has a negative eigenvalue.

Model of the linear theorem: ``h_t = A h_{t-1} + x_t``, full read-out, ``theta = vec(A)``.  For directions
``E'``, ``E`` the traces ``z_t = D z_{t-1} + E' h_{t-1}`` (SnAp-1, ``D = diag A``) and
``v_t = A v_{t-1} + E h_{t-1}`` (exact) are jointly stationary with ``h``, and ``<E', C E> = tr E[v z^T]``,
where

    Pi  = A Pi A^T + I,
    N_v = A N_v A^T + E Pi A^T          (N_v = E[v h^T]),
    N_z = A N_z D^T + A Pi E'^T         (N_z = E[h z^T]),
    M   = A M D^T + A N_v E'^T + E N_z D^T + E Pi E'^T.

All equations are linear with rational coefficients and are solved in exact arithmetic (SymPy).

    python -m experiments.linear_dynamics.snap1_exact_instability
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.common import make_parser, write_json

RESULTS_SUBDIR = ("linear_dynamics",)
OUTPUT_NAME = "snap1_exact_instability.json"

A = sp.Matrix([[-sp.Rational(1, 4), -sp.Rational(1, 4), 3], [0, -sp.Rational(1, 2), 3], [3, -3, 0]])


def vec(M):
    return sp.Matrix([M[i, j] for i in range(M.rows) for j in range(M.cols)])


def unvec(v, n):
    return sp.Matrix(n, n, list(v))


def kron(M, N):
    """``vec(M X N^T) = kron(M, N) vec X`` with rows stacked."""
    return sp.kronecker_product(M, N)


def cross_gram(A, D):
    """Exact cross-Gram of the trace with propagation matrix ``D``, and the state covariance ``Pi``."""
    n = A.rows
    I = sp.eye(n * n)
    L1 = (I - kron(A, A)).inv()
    L2 = (I - kron(A, D)).inv()
    Pi = unvec(L1 * vec(sp.eye(n)), n)
    basis = [unvec(sp.eye(n * n)[:, i], n) for i in range(n * n)]
    Nv = [unvec(L1 * vec(E * Pi * A.T), n) for E in basis]
    Nz = [unvec(L2 * vec(A * Pi * E.T), n) for E in basis]
    C = sp.zeros(n * n, n * n)
    for a, Ea in enumerate(basis):
        for b, Eb in enumerate(basis):
            C[a, b] = unvec(L2 * vec(A * Nv[b] * Ea.T + Eb * Nz[a] * D.T + Eb * Pi * Ea.T), n).trace()
    return C, Pi


def numeric_spectrum(M):
    """Eigenvalues in floating point, sorted by real part (for display only)."""
    values = np.linalg.eigvals(np.array(M.tolist(), dtype=float))
    return sorted([complex(z) for z in values], key=lambda z: z.real)


def main(argv=None):
    parser = make_parser(__doc__, RESULTS_SUBDIR, device=None)
    args = parser.parse_args(argv)
    n = A.rows
    lam = sp.symbols("lambda")
    characteristic = sp.factor(A.charpoly(lam).as_expr())
    C_snap1, Pi = cross_gram(A, sp.diag(*[A[i, i] for i in range(n)]))
    C_immediate, _ = cross_gram(A, sp.zeros(n, n))  # immediate rule: D = 0
    C_exact, _ = cross_gram(A, A)  # exact: must be symmetric positive definite
    # closed form of the linear theorem for the immediate rule
    closed_form = (sp.eye(n * n) - kron(A, A)).inv() * kron(sp.eye(n), Pi)
    assert sp.simplify(C_immediate - closed_form) == sp.zeros(n * n, n * n)
    assert C_exact == C_exact.T and C_exact.is_positive_definite
    det_snap1, det_immediate = C_snap1.det(), C_immediate.det()
    eig_snap1, eig_immediate = numeric_spectrum(C_snap1), numeric_spectrum(C_immediate)
    record = dict(
        matrix=[[str(x) for x in A.row(i)] for i in range(n)],
        characteristic_polynomial=str(characteristic),
        spectral_radius=float(max(abs(z) for z in numeric_spectrum(A))),
        det_snap1=str(det_snap1),
        det_snap1_float=float(det_snap1),
        det_immediate=str(det_immediate),
        det_immediate_float=float(det_immediate),
        eigenvalues_snap1=[[z.real, z.imag] for z in eig_snap1],
        eigenvalues_immediate=[[z.real, z.imag] for z in eig_immediate],
        gauss_newton_max_eig=float(max(z.real for z in numeric_spectrum(C_exact))),
    )
    print("A =", A.tolist())
    print("characteristic polynomial:", characteristic, " spectral radius", record["spectral_radius"])
    print("det C_snap1 =", det_snap1, "=", float(det_snap1), "(negative: an odd number of negative real eigenvalues)")
    print("eigenvalues of C_snap1:", ", ".join(f"{z.real:.4g}{z.imag:+.3g}i" for z in eig_snap1))
    print("eigenvalues of C_immediate:", ", ".join(f"{z.real:.4g}{z.imag:+.3g}i" for z in eig_immediate))
    write_json(args.output_dir / OUTPUT_NAME, record, indent=1)


if __name__ == "__main__":
    main()
