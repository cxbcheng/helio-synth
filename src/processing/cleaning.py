import numpy as np


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) index pairs (stop exclusive) for each contiguous run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[0::2], edges[1::2]))


def interpolate_short_gaps(velocities: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Linearly interpolates over NaN runs of length <= max_gap samples.
    Longer runs (and any NaNs with no valid data on one side, e.g. at the
    array's edges) are left as NaN. Assumes `velocities` is on a uniform
    time grid.
    In practice, short runs are safe to interpolate without significantly
    compromising signal frequencies. For longer runs, more care should be
    taken to handle missing data without distorting the signal. One good choice
    is to segment the output of this method interpolation and filter on each
    segment, then zero-filling NaN values after joining segments.
    Note that the input array is not modified.
    :param velocities: Uniform-cadence velocity data.
    :param max_gap: Maximum length run of NaNs to interpolate. Default is 3
        (2.25 minutes for a 45-second cadence).
    """
    v = velocities.copy()
    is_nan = np.isnan(v)
    if not is_nan.any():
        return v

    for start, stop in find_runs(is_nan):
        if stop - start > max_gap or start == 0 or stop == len(v):
            continue  # leave long / edge gaps as NaN
        v[start:stop] = np.interp(np.arange(start, stop), [start - 1, stop], [v[start - 1], v[stop]])
    return v