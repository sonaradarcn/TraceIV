"""Build ``data/household_power_segments.txt`` from the UCI household power record.

Source: "Individual household electric power consumption" (Hebrail and Berard, UCI Machine Learning
Repository, doi:10.24432/C58K54, CC BY 4.0): global active power of one household in kW, one value per
minute from December 2006 to November 2010.

    python data/prepare_household_power.py path/to/household_power_consumption.txt

Processing.  Gaps of at most 60 minutes are interpolated linearly; the record is averaged over
consecutive 10-minute bins; the bins are scanned from the start for disjoint windows of 8000 bins (about
55 days) that contain no longer gap, and the first 20 windows are written, one per line.  Seed ``s`` of
the task ``power`` uses line ``s``, so no two seeds share an observation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MAX_INTERPOLATED_GAP = 60  # minutes
BIN = 10  # minutes
SEGMENT_LENGTH = 8000  # bins
NUM_SEGMENTS = 20
OUTPUT = Path(__file__).resolve().parent / "household_power_segments.txt"


def read_global_active_power(path):
    values = []
    with open(path) as handle:
        next(handle)
        for line in handle:
            field = line.split(";")[2].strip()
            values.append(float("nan") if field in ("?", "") else float(field))
    return np.array(values)


def segments(power):
    n = len(power)
    missing = np.isnan(power)
    index = np.arange(n)
    filled = np.interp(index, index[~missing], power[~missing])
    start = 0
    while start < n:  # leave gaps longer than the limit missing
        if not missing[start]:
            start += 1
            continue
        stop = start
        while stop < n and missing[stop]:
            stop += 1
        if stop - start > MAX_INTERPOLATED_GAP:
            filled[start:stop] = np.nan
        start = stop
    num_bins = n // BIN
    bins = filled[: num_bins * BIN].reshape(num_bins, BIN).mean(1)
    bad = np.isnan(bins)
    out, start = [], 0
    while start + SEGMENT_LENGTH <= num_bins and len(out) < NUM_SEGMENTS:
        window = bad[start : start + SEGMENT_LENGTH]
        if window.any():
            start += int(np.nonzero(window)[0].max()) + 1
        else:
            out.append(bins[start : start + SEGMENT_LENGTH])
            start += SEGMENT_LENGTH
    if len(out) < NUM_SEGMENTS:
        raise SystemExit(f"only {len(out)} gap-free segments found")
    return np.stack(out)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    np.savetxt(OUTPUT, segments(read_global_active_power(sys.argv[1])), fmt="%.4f")
    print(f"wrote {OUTPUT.name}: {NUM_SEGMENTS} segments of {SEGMENT_LENGTH} bins")
