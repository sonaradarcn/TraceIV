"""Every number quoted in the paper, recomputed from the result files.

python -m analysis.paper_numbers                 # all sections
python -m analysis.paper_numbers restoration     # one section
"""

from __future__ import annotations

import argparse

from analysis import (
    calibrated_point,
    linear_dynamics,
    misspecification,
    nonstationarity,
    prediction_tasks,
    relevance,
    restoration,
)

SECTIONS = {
    "linear_dynamics": linear_dynamics.report,
    "calibrated_point": calibrated_point.report,
    "relevance": relevance.report,
    "misspecification": misspecification.report,
    "restoration": restoration.report,
    "prediction_tasks": prediction_tasks.report,
    "nonstationarity": nonstationarity.report,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sections", nargs="*", help="sections to report (default: all): " + ", ".join(SECTIONS))
    args = parser.parse_args(argv)
    unknown = [name for name in args.sections if name not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s) {', '.join(unknown)}; choose from {', '.join(SECTIONS)}")
    for name in args.sections or SECTIONS:
        SECTIONS[name]()


if __name__ == "__main__":
    main()
