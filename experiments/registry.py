"""Map from the experiments of the paper to their entry points and result files.

``EXPERIMENTS`` is the single place that ties together, for every experiment,

* the module that runs it (``python -m <module>``),
* the directory of its result files below the results root,
* where it appears in the paper (section, figures, tables), and
* the identifier it carried during development (``E1`` ... ``E9``), kept so that earlier notes and review
  logs can still be matched to the code; experiments added in revision have none.

``python -m experiments.registry`` prints the table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experiment:
    name: str  # key used by `experiments.reproduce` and in configs/paper_runs.json
    module: str  # entry point
    results_subdir: str  # below the results root
    paper_section: str
    paper_items: str  # figures and tables that use the results
    development_id: str
    summary: str


EXPERIMENTS = [
    Experiment(
        "closed_form_checks",
        "experiments.linear_dynamics.closed_form_checks",
        "linear_dynamics",
        "9.1 Linear dynamics: the closed forms",
        "Fig. 4a-c",
        "E1",
        "closed forms of the linear theorem; non-normality sweep at fixed spectrum",
    ),
    Experiment(
        "snap1_exact_instability",
        "experiments.linear_dynamics.snap1_exact_instability",
        "linear_dynamics",
        "9.1 Linear dynamics: the closed forms",
        "Fig. 4d",
        "E1c",
        "exact 3x3 system in which the SnAp-1 cross-Gram has a negative eigenvalue (rational arithmetic)",
    ),
    Experiment(
        "stationary_risk",
        "experiments.calibrated_point.stationary_risk",
        "calibrated_point/stationary_risk",
        "9.2 Stability and efficiency at a calibrated point; 9.4 Gated cells",
        "Fig. 6a,b,d; Table 2",
        "E2 (tanh), E2g (gated)",
        "stationary excess risk at a calibrated point against the Lyapunov prediction",
    ),
    Experiment(
        "efficiency_constant",
        "experiments.calibrated_point.efficiency_constant",
        "calibrated_point/efficiency_constant",
        "9.2 Stability and efficiency at a calibrated point",
        "Fig. 6c",
        "E2b",
        "asymptotic constant of the averaged iterate against the sum of inverse squared canonical correlations",
    ),
    Experiment(
        "readout_rank",
        "experiments.relevance.readout_rank",
        "relevance/readout_rank",
        "9.3 Relevance: what a sparse read-out removes; 9.4 Gated cells",
        "Table 1; Table 2",
        "E3 (tanh), E3g (gated)",
        "rank of the cross-Gram and excess risk when m of n units are read out",
    ),
    Experiment(
        "displacement",
        "experiments.misspecification.displacement",
        "misspecification/displacement",
        "9.5 Displacement under misspecification",
        "Fig. 7",
        "E4",
        "displacement of the limit and risk gap of a misspecified student against the first-order prediction",
    ),
    Experiment(
        "stability_threshold",
        "experiments.restoration.stability_threshold",
        "restoration/stability_threshold",
        "9.1 (SnAp-1 drift); 9.6 Restoration: stability threshold and exchange rate",
        "Fig. 5",
        "E5a",
        "SnAp-1 leaving a calibrated point of a linear system; restoration above and below the threshold",
    ),
    Experiment(
        "exchange_rate",
        "experiments.restoration.exchange_rate",
        "restoration/exchange_rate",
        "9.6 Restoration: stability threshold and exchange rate",
        "text",
        "E5b",
        "error of the projected bias-field estimate against the exchange-rate law",
    ),
    Experiment(
        "cost_frontier",
        "experiments.restoration.cost_frontier",
        "restoration/cost_frontier",
        "9.6 Restoration: stability threshold and exchange rate",
        "Fig. 8a-c",
        "E8",
        "excess risk against arithmetic per step for restoration by tangent probes",
    ),
    Experiment(
        "step_cost_benchmark",
        "experiments.restoration.step_cost_benchmark",
        "restoration/step_cost",
        "9.6 Restoration: stability threshold and exchange rate",
        "Fig. 8d",
        "bench",
        "time per step and trace memory of the recursions",
    ),
    Experiment(
        "local_regime",
        "experiments.calibrated_point.local_regime",
        "calibrated_point/local_regime",
        "9.7 Away from the calibrated point",
        "text",
        "E9",
        "excess risk of every rule started at increasing distances from a calibrated point",
    ),
    Experiment(
        "online_prediction",
        "experiments.prediction_tasks.online_prediction",
        "prediction_tasks",
        "9.8 Prediction tasks with a learned read-out",
        "Tables 3 and 4; Fig. 9",
        "E7",
        "prediction tasks from a random start and after handover from RTRL",
    ),
    Experiment(
        "tracking_lag",
        "experiments.nonstationarity.tracking_lag",
        "nonstationarity/tracking_lag",
        "9.9 A drifting teacher",
        "Table 5",
        "added in revision",
        "lag of every rule behind a teacher whose parameters drift, against the tracking proposition",
    ),
]

BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}


def main():
    for e in EXPERIMENTS:
        print(f"{e.name}\n    module    python -m {e.module}\n    results   results/{e.results_subdir}/")
        print(
            f"    paper     Sec. {e.paper_section}  ({e.paper_items})\n    was       {e.development_id}\n    {e.summary}\n"
        )


if __name__ == "__main__":
    main()
