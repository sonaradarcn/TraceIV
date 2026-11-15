"""Section 9.1, linear dynamics: checks of the closed forms and the exact SnAp-1 example."""

from __future__ import annotations

from analysis.common import load_optional


def report_theorem_checks():
    print("== Linear dynamics: checks of the closed forms on random stable matrices")
    record = load_optional("linear_dynamics", "closed_forms.json")
    if record is None:
        return
    checks = record["theorem_checks"]
    within_unit_radius = [c for c in checks if c["numerical_radius"] <= 1]
    print(
        f"  n={len(checks)} formula err max {max(c['closed_form_error'] for c in checks):.1e}; "
        f"criterion agrees {sum(c['criterion_agrees'] for c in checks)}/{len(checks)}; "
        f"spectrum err max {max(c['spectrum_error'] for c in checks):.1e}; "
        f"w<=1 and sym>0: {sum(c['sym_positive'] for c in within_unit_radius)}/{len(within_unit_radius)}; "
        f"sym>0 overall {sum(c['sym_positive'] for c in checks)}/{len(checks)}"
    )


def report_exact_example():
    print("\n== Linear dynamics: exact SnAp-1 example ==")
    d = load_optional("linear_dynamics", "snap1_exact_instability.json")
    if d is None:
        return
    print(
        f"A={d['matrix']} charpoly {d['characteristic_polynomial']} det C_snap1 = {d['det_snap1']} "
        f"({d['det_snap1_float']:.4g}); most negative eigenvalue {d['eigenvalues_snap1'][0][0]:.4g}; "
        f"immediate rule: eigenvalues in [{d['eigenvalues_immediate'][0][0]:.4g}, "
        f"{d['eigenvalues_immediate'][-1][0]:.4g}], lambda_max(G)={d['gauss_newton_max_eig']:.4g}"
    )


def report():
    report_theorem_checks()
    report_exact_example()


if __name__ == "__main__":
    report()
