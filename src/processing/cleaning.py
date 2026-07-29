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


def _raised_cosine_ramp(a: float, b: float, n: int) -> np.ndarray:
    """
    Returns n samples interpolating smoothly from `a` to `b` via raised cosine.
    The motivation comes from this function having derivative zero at both endpoints.
    """
    t = np.linspace(0, 1, n)
    w = (1 - np.cos(np.pi * t)) / 2
    return a + (b - a) * w


def zero_fill_with_taper(
        velocities: np.ndarray,
        taper_frames: int = 10,
        fill_value: float = 0.0,
) -> np.ndarray:
    """
    Replaces remaining NaN runs in a signal with fill_value, ramping
    smoothly in from each edge's last valid filtered sample rather than
    stepping directly to fill_value. Eliminates the audible clicks a hard
    discontinuity would cause at gap boundaries.

    Meant to run after bandpass filtering (e.g. after filter_segments),
    on filtered/band-limited output only; fill_value=0.0 approximates
    "no oscillation," which is a safe assumption for zero-mean, band-limited
    p-mode data but would not be a neutral choice on raw velocity.

    A NaN run touching either edge of the array (no valid neighbor on that
    side) is ramped only from the side that has one; if neither side has a
    valid neighbor, the run is filled flat at fill_value with no ramp.

    :param velocities: Bandpass-filtered signal, possibly containing NaN runs
        (from long gaps left by filter_segments, or segments too short to filter).
    :param taper_frames: Ramp length in samples at each edge of a gap.
        Gaps shorter than 2*taper_frames use a proportionally shorter ramp
        (min(taper_frames, gap_length // 2)).
    :param fill_value: Value the interior of each gap settles to. Default is 0.0 (silence/rest).
    :return: Signal of the same length as `filtered`, with no NaNs.
    """
    v = velocities.copy()
    is_nan = np.isnan(v)
    if not is_nan.any():
        return v

    for start, stop in find_runs(is_nan):
        length = stop - start
        ramp = min(taper_frames, length // 2) if length >= 2 else 0
        left_val = v[start - 1] if start > 0 else None
        right_val = v[stop] if stop < len(v) else None

        gap = np.full(length, fill_value, dtype=np.float64)
        if left_val is not None and ramp > 0:
            gap[:ramp] = _raised_cosine_ramp(left_val, fill_value, ramp)
        if right_val is not None and ramp > 0:
            gap[-ramp:] = _raised_cosine_ramp(fill_value, right_val, ramp)

        v[start:stop] = gap

    return v