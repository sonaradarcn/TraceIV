"""Section 9.5: displacement of the limit and risk gap under misspecification."""

from __future__ import annotations

import numpy as np

from analysis.common import load_all


def report():
    print("== Misspecification: displacement of the limit")
    records = sorted(load_all("misspecification", "displacement"), key=lambda d: d["coupling"])
    if not records:
        return

    def length_ratio(r):
        return np.array(r["displacement"]) / np.array(r["predicted_displacement"])

    for d in records:
        eps = d["coupling"]
        print(
            f"  eps={eps} reference point: |grad L| {np.median(d['reference_gradient_norm']):.2e}, "
            f"Newton decrement {np.median(d['newton_decrement']):.2e}, "
            f"min eig Hessian {np.median(d['hessian_min_eig']):.2e}, "
            f"|GN-Hessian|/|Hessian| {np.median(d['gauss_newton_vs_hessian']):.3f}"
        )
        for rule, r in d["rules"].items():
            gram_ratio = np.array(r["displacement"]) / np.array(r["predicted_displacement_cross_gram"])
            print(
                f"      {rule:9s} |DV-C|/|DV| {np.median(r['jacobian_vs_cross_gram']):.3f}; "
                f"with the cross-Gram in place of DV: cos {np.median(r['direction_cosine_cross_gram']):.3f} "
                f"ratio {np.median(gram_ratio):.2f}"
            )
            print(
                f"  eps={eps} {rule:9s} |disp| {np.median(r['displacement']):.3e} "
                f"pred {np.median(r['predicted_displacement']):.3e} ratio {np.median(length_ratio(r)):.2f} "
                f"cos {np.median(r['direction_cosine']):.3f} gap {np.median(r['risk_gap']):.2e} "
                f"pred {np.median(r['predicted_risk_gap']):.2e}"
            )
    cosines = np.array([r["direction_cosine"] for d in records for r in d["rules"].values()])
    lengths = np.array([length_ratio(r) for d in records for r in d["rules"].values()])
    print(
        f"  all {cosines.size} (system, rule, eps): cos>0.95 in {(cosines > 0.95).sum()}, "
        f"cos>0.9 in {(cosines > 0.9).sum()}, min cos {cosines.min():.3f}; "
        f"length within 10% in {(np.abs(lengths - 1) < 0.1).sum()}, "
        f"worst length ratio {lengths.max():.2f}/{lengths.min():.2f}"
    )
    for d in records:
        newton = np.array(d["newton_decrement"])
        worst = max((newton / np.array(r["risk_gap"])).max() for r in d["rules"].values())
        print(
            f"  eps={d['coupling']} per system: min eig Hessian {np.round(d['hessian_min_eig'], 4).tolist()}; "
            f"max over rules and systems of Newton decrement / measured gap {worst:.3f}"
        )
        for rule, r in d["rules"].items():
            gap_ratio = np.array(r["risk_gap"]) / np.array(r["predicted_risk_gap"])
            print(
                f"      {rule:9s} cos per system {np.round(r['direction_cosine'], 3).tolist()} "
                f"length ratio {np.round(length_ratio(r), 2).tolist()} gap ratio {np.round(gap_ratio, 2).tolist()}"
            )
    log_eps = np.log([d["coupling"] for d in records])
    for rule in records[0]["rules"]:
        displacement = np.log([np.median(d["rules"][rule]["displacement"]) for d in records])
        gap = np.log([np.median(d["rules"][rule]["risk_gap"]) for d in records])
        print(
            f"  slope log|disp| vs log eps ({rule}): {np.polyfit(log_eps, displacement, 1)[0]:.2f}; "
            f"slope log gap: {np.polyfit(log_eps, gap, 1)[0]:.2f}"
        )


if __name__ == "__main__":
    report()
