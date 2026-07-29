import numpy as np
from numpy import ndarray
from scipy import signal

from src.heliosynth.processing.cleaning import find_runs


def _get_min_signal_len(filter_order: int) -> int:
    """
    scipy.signal.sosfiltfilt needs roughly 3x the combined filter length as padding;
    a signal shorter than that raises a scipy error. Thus, this method is to aid in
    input validation by getting the minimum signal length required for a filter order.
    Computed as 3 * (2 * filter_order + 1).
    """
    return 3 * (2 * filter_order + 1)


def bandpass_filter(
        x: ndarray,
        low_cutoff: float = 0.001,
        high_cutoff: float = 0.005,
        fs: float = 1/45,
        filter_order: int = 2,
) -> ndarray:
    """
    Applies a forward-backward Butterworth bandpass filter to a signal.

    Assumes x is a complete, uniformly-sampled, finite signal. Ensure that
    the signal's gaps (NaN and bad values) are handled prior to applying
    this filter. Typically under scipy.signal.sosfiltfilt, a single NaN/Inf
    in x silently corrupts the entire output (since IIR filters propagate NaN
    through their whole recursive history in both directions). Hence, the
    purpose of this function is to raises instead of returning a corrupted result.

    For filtering data with NaN runs, see `filter_segments`.

    :param x: 1-D array of the signal to filter. Must be finite (no NaN/Inf).
    :param low_cutoff: Low cutoff frequency (Hz) for the bandpass filter.
    :param high_cutoff: High cutoff frequency (Hz) for the bandpass filter.
    :param fs: Sampling frequency (Hz). Defaults to 1/45 Hz (HMI's 45s cadence).
    :param filter_order: Butterworth order before forward-backward filtering;
        effective order after sosfiltfilt is double this.
    :return: Filtered signal, same length as x.
    :raises ValueError: if x contains NaN/Inf, if cutoffs are invalid relative
        to the Nyquist frequency, or if x is too short to filter stably.
    """
    x = np.asarray(x, dtype=np.float64)

    # Input validation
    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        n_bad = np.count_nonzero(~np.isfinite(x))
        raise ValueError(f"x contains {n_bad} NaN/Inf value(s). Fill or segment gaps before calling bandpass_filter.")

    nyquist = fs / 2
    if not (0 < low_cutoff < high_cutoff < nyquist):
        raise ValueError(f"Require 0 < low_cutoff ({low_cutoff}) < high_cutoff, ({high_cutoff}) < nyquist ({nyquist:.5f}) for fs={fs}")

    min_len = _get_min_signal_len(filter_order)
    if len(x) < min_len:
        raise ValueError(f"x has {len(x)} samples; need at least ~{min_len} to filter stably at filter_order={filter_order}.")

    sos = signal.butter(N=filter_order, Wn=[low_cutoff, high_cutoff],
                        fs=fs, btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, x)


def filter_segments(
        velocities: np.ndarray,
        low_cutoff: float,
        high_cutoff: float,
        fs: float = 1/45,
        filter_order: int = 2
) -> np.ndarray:
    """
    Filters contiguous non-NaN segments using `bandpass_filter`.
    Segments shorter than min_len = 3 * (2 * filter_order + 1)
    are safely left as NaN.
    """
    v_filtered = np.full_like(velocities, np.nan)
    is_valid = np.isfinite(velocities)
    min_len = _get_min_signal_len(filter_order)

    for start, stop in find_runs(is_valid):
        segment = velocities[start:stop]

        if len(segment) >= min_len:
            v_filtered[start:stop] = bandpass_filter(
                segment,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                fs=fs,
                filter_order=filter_order
            )

    return v_filtered