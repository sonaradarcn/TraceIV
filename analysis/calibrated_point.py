"""Sections 9.2, 9.4 and 9.7: stationary risk and efficiency at a calibrated point, and its neighbourhood."""

from __future__ import annotations

import numpy as np
from scipy import stats

from analysis.common import load, load_by_family, result_files
from trace_instruments import rules

STATIONARY_RISK = ("calibrated_point", "stationary_risk")


def report_stationary_risk():
    """Tanh cell, Gamma = G^-1: measured against predicted stationary excess risk."""
    print("== Calibrated point (tanh): measured / predicted stationary excess risk")
    rows = {}
    for d in load_by_family(*STATIONARY_RISK, pattern="tanh_*.json"):
        for rule, r in d["rules"].items():
            ok = ~np.array(r["diverged"])
            ratio = np.array(r["measured_excess_risk"])[ok] / np.array(r["predicted_excess_risk"])[ok]
            rows.setdefault(rule, []).append(
                dict(
                    family=d["family"],
                    gain=d["gain"],
                    ratio=ratio,
                    num_diverged=int((~ok).sum()),
                    efficiency=np.array(r["efficiency_loss"])[ok],
                    cosine=np.array(r["gradient_cosine"])[ok],
                    sym_min=np.array(r["sym_min_eig"])[ok],
                    preconditioned_min=np.array(r["min_real_eig_preconditioned"])[ok],
                )
            )
    for rule, entries in rows.items():
        ratios = np.concatenate([e["ratio"] for e in entries])
        print(
            f"  {rule:9s} ratio median {np.nanmedian(ratios):.3f} "
            f"IQR [{np.nanpercentile(ratios, 25):.3f},{np.nanpercentile(ratios, 75):.3f}] "
            f"nan={int(np.isnan(ratios).sum())} n={len(ratios)} diverged={sum(e['num_diverged'] for e in entries)} "
            f"| sym(C)min<0 in {int(sum((e['sym_min'] < 0).sum() for e in entries))}, "
            f"minRe(GamC)<0 in {int(sum((e['preconditioned_min'] < 0).sum() for e in entries))}"
        )
        for e in entries:

            def median_or_nan(x):
                return np.median(x) if len(x) else float("nan")

            ratio = np.nanmedian(e["ratio"]) if len(e["ratio"]) else float("nan")
            print(
                f"       {e['family']:8s} g={e['gain']}: ratio {ratio:.3f} eff {median_or_nan(e['efficiency']):.2f} "
                f"cos {median_or_nan(e['cosine']):.2f} div {e['num_diverged']}"
            )
    approximate = [entries for rule, entries in rows.items() if rule != rules.RTRL]
    if not approximate:
        return
    efficiency = np.concatenate([np.concatenate([e["efficiency"] for e in entries]) for entries in approximate])
    cosine = np.concatenate([np.concatenate([e["cosine"] for e in entries]) for entries in approximate])
    ok = np.isfinite(efficiency) & (efficiency < 1e3)
    print(
        f"  Spearman(cos, eff) over non-rtrl rules = {stats.spearmanr(cosine[ok], efficiency[ok]).statistic:.3f} "
        f"(n={ok.sum()}; {int((~ok).sum())} near-singular excluded)"
    )


def report_efficiency_constant():
    print("== Calibrated point: efficiency constant (averaged iterates)")
    for d in load_by_family("calibrated_point", "efficiency_constant"):
        for rule, r in d["rules"].items():
            ratio = np.array(r["measured_constant"]) / np.array(r["predicted_constant"])
            P = d["num_parameters"]
            print(
                f"  {d['family']:8s} {rule:9s} pred/P {np.median(r['predicted_constant']) / P:.3f} "
                f"meas/P {np.median(r['measured_constant']) / P:.3f} ratio per system {np.round(ratio, 3).tolist()} "
                f"z {np.round(r['mean_displacement_z'], 2).tolist()} div {r['diverged']}"
            )


def report_gated_stationary_risk():
    print("\n== Calibrated point (gated cells, plain constant step, Gamma = I) ==")
    pooled = {}
    for path in result_files(*STATIONARY_RISK, pattern="gru_*.json"):
        d = load(path)
        for rule, r in d["rules"].items():
            measured = np.array(r["measured_excess_risk"], float)
            predicted = np.array(r["predicted_excess_risk"], float)
            preconditioned_min = np.array(r["min_real_eig_preconditioned"])
            diverged = np.array(r["diverged"])
            pooled.setdefault(rule, []).append(
                (measured / predicted, diverged, np.array([min(x) for x in r["coherence"]]))
            )
            print(
                f"{path.stem:28s} {rule:9s} stable(pred) {int((preconditioned_min > 0).sum())}/{preconditioned_min.size} "
                f"stayed {int((~diverged).sum())}/{diverged.size} "
                f"meas/pred excess median {np.nanmedian(measured / predicted):.3f} "
                f"[{np.nanmin(measured / predicted):.2f}, {np.nanmax(measured / predicted):.2f}] "
                f"sym(C) indefinite in {int((np.array(r['sym_min_eig']) < 0).sum())} "
                f"cos {np.median(r['gradient_cosine']):.2f} eff loss {np.median(r['efficiency_loss']):.2f}"
            )
    for rule, entries in pooled.items():
        ratios = np.concatenate([e[0] for e in entries])
        print(
            f"   all families {rule:9s} meas/pred median {np.nanmedian(ratios):.3f} "
            f"IQR [{np.nanpercentile(ratios, 25):.3f}, {np.nanpercentile(ratios, 75):.3f}] "
            f"diverged {int(np.concatenate([e[1] for e in entries]).sum())}/{ratios.size} "
            f"rho2min median {np.median(np.concatenate([e[2] for e in entries])):.3f}"
        )


def report_local_regime():
    print(
        "\n== Away from the calibrated point (excess risk of the averaged iterate; "
        "ok = within 5x of RTRL started at the smallest radius) =="
    )
    for path in result_files("calibrated_point", "local_regime"):
        d = load(path)
        excess = np.array(d["excess_risk"], float)
        reference = np.nanmedian(excess[:, 0, 0])
        S = excess.shape[0]
        print(f"-- {path.stem} P={d['num_parameters']} alpha={d['args']['step_size']} T={d['args']['learning_steps']}")
        for i, radius in enumerate(d["radii"]):
            print(
                f"   r={radius!s:5s} start {np.median(np.array(d['excess_risk_start'])[:, i]):.2e} | "
                + " | ".join(
                    f"{rule} {np.nanmedian(excess[:, i, j]):.2e} "
                    f"ok {int(np.nansum(excess[:, i, j] < 5 * reference))}/{S} "
                    f"div {int(np.isnan(excess[:, i, j]).sum())}"
                    + (f" ratio {np.nanmedian(excess[:, i, j] / excess[:, i, 0]):.2f}" if j else "")
                    for j, rule in enumerate(d["rules"])
                )
            )


def report():
    report_stationary_risk()
    report_efficiency_constant()
    report_gated_stationary_risk()
    report_local_regime()


if __name__ == "__main__":
    report()
