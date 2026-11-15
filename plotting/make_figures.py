"""Draw the data figures of the paper from the result files.

    python -m plotting.make_figures                    # all figures, into figures/
    python -m plotting.make_figures wellspec tasks     # some of them
    python -m plotting.make_figures --output-dir <dir>

Figure     Paper     Module                        Result files
nonnormal  Fig. 4    plotting.nonnormality         linear_dynamics/
stabilise  Fig. 5    plotting.stability_threshold  restoration/stability_threshold/
wellspec   Fig. 6    plotting.calibrated_point     calibrated_point/{stationary_risk,efficiency_constant}/
misspec    Fig. 7    plotting.misspecification     misspecification/displacement/
frontier   Fig. 8    plotting.cost_frontier        restoration/{cost_frontier,step_cost}/
tasks      Fig. 9    plotting.prediction_tasks     prediction_tasks/
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from plotting import calibrated_point, cost_frontier, misspecification, nonnormality, prediction_tasks
from plotting import stability_threshold

FIGURES = {
    module.NAME: module.draw
    for module in (
        nonnormality,
        stability_threshold,
        calibrated_point,
        misspecification,
        cost_frontier,
        prediction_tasks,
    )
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figures", nargs="*", help="figures to draw (default: all): " + ", ".join(FIGURES))
    parser.add_argument("--output-dir", help="where to write the figures (default: figures/)")
    args = parser.parse_args(argv)
    unknown = [name for name in args.figures if name not in FIGURES]
    if unknown:
        parser.error(f"unknown figure(s) {', '.join(unknown)}; choose from {', '.join(FIGURES)}")
    if args.output_dir:
        os.environ["TRACE_INSTRUMENTS_FIGURES_DIR"] = args.output_dir
    failed = []
    for name in args.figures or FIGURES:
        try:
            FIGURES[name]()
        except Exception:  # keep drawing the other figures; report at the end
            traceback.print_exc()
            failed.append(name)
    if failed:
        sys.exit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
