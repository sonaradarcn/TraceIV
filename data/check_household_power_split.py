"""Does any interpolated gap touch the training/held-out boundary of a chosen window?

usage: python data/check_household_power_split.py <raw household_power_consumption.txt> <household_power_segments.txt>
Repeats the segmentation of prepare_household_power.py, checks that it reproduces the shipped file, and
reports, for each window, the missing minutes in the bins on either side of the split.
"""

import sys

import numpy as np

MAX_GAP, BIN, LENGTH, NUM = 60, 10, 8000, 20
values = []
with open(sys.argv[1]) as handle:
    next(handle)
    for line in handle:
        field = line.split(";")[2].strip()
        values.append(float("nan") if field in ("?", "") else float(field))
power = np.array(values)
n = len(power)
missing = np.isnan(power)
index = np.arange(n)
filled = np.interp(index, index[~missing], power[~missing])
start = 0
while start < n:
    if not missing[start]:
        start += 1
        continue
    stop = start
    while stop < n and missing[stop]:
        stop += 1
    if stop - start > MAX_GAP:
        filled[start:stop] = np.nan
    start = stop
num_bins = n // BIN
bins = filled[: num_bins * BIN].reshape(num_bins, BIN).mean(1)
bad = np.isnan(bins)
starts, start = [], 0
while start + LENGTH <= num_bins and len(starts) < NUM:
    window = bad[start : start + LENGTH]
    if window.any():
        start += int(np.nonzero(window)[0].max()) + 1
    else:
        starts.append(start)
        start += LENGTH
shipped = np.loadtxt(sys.argv[2])
rebuilt = np.stack([bins[s : s + LENGTH] for s in starts])
same = bool(np.allclose(np.round(rebuilt, 4), shipped, rtol=0, atol=1e-9))
print("rebuilt equals shipped file to 4 decimals:", same)
disjoint = all(b - a >= LENGTH for a, b in zip(starts, starts[1:]))
print("windows are disjoint:", disjoint)
print("window starts (bin index):", starts)
cut = int(0.8 * (LENGTH - 1))  # pairs (u_t, y_t) = (s_t, s_{t+1}); training pairs 0..cut-1 use bins 0..cut
total_missing = 0
straddle = 0
for k, s in enumerate(starts):
    minutes = missing[s * BIN : (s + LENGTH) * BIN]
    total_missing += int(minutes.sum())
    # last bin seen in training is `cut` (as a target); first held-out-only bin is cut + 1
    boundary_minute = (s + cut + 1) * BIN
    # a gap straddles the boundary if the minute before and the minute after it are both missing
    if missing[boundary_minute - 1] and missing[boundary_minute]:
        straddle += 1
        print("window", k, "has an interpolated gap across the split")
    near = int(missing[boundary_minute - 60 : boundary_minute + 60].sum())
    if near:
        print("window", k, "missing minutes within an hour of the split:", near)
print("interpolated minutes in the 20 windows:", total_missing, "of", NUM * LENGTH * BIN)
print("windows with an interpolated gap across the split:", straddle)
if not (same and disjoint and straddle == 0):
    sys.exit("FAILED")
print("OK")
