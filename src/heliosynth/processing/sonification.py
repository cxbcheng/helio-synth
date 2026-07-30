import numpy as np
from scipy import signal

# Safety ceiling: 10 minutes @ 48 kHz (4.8M samples)
MAX_OUTPUT_SAMPLES = 10 * 60 * 48000


def resample_and_compress(
    x: np.ndarray,
    f_orig: float,
    f_target: int = 8000,
    alpha: float = 40000,
    max_output_samples: int = MAX_OUTPUT_SAMPLES,
) -> np.ndarray:
    """
    Time-scales and resamples a 1D signal to a target audio sample rate.

    Applies a time-compression factor `alpha` to shift infrasonic solar oscillations
    into the human audible frequency spectrum [20 Hz, 20000 Hz].

    :param x: The filtered, gap-free signal to be resampled and compressed.
        Must contain no NaN/Inf.
    :param f_orig: Sampling frequency (Hz) of x, i.e. 1 / actual data cadence.
    :param f_target: Target sampling frequency of the compressed audio (Hz).
    :param alpha: Time-scaling compression factor.
    :param max_output_samples: Safety ceiling on output length. Guards against
        runaway memory/CPU use in signal.resample when alpha/f_orig/f_target are
        set inconsistently. Default is generous (10 minutes at 8kHz).
    :return: The resampled and compressed audio.
    :raises ValueError: if any frequency/alpha parameter is non-positive, if
        x contains non-finite values, or if the computed output length is
        outside [2, max_output_samples].
    """
    # Higher precision and type checking
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN/Inf. Fill missing data with zero_fill_with_taper (or equivalent) first")
    if f_orig <= 0 or f_target <= 0 or alpha <= 0:
        raise ValueError("f_orig, f_target, and alpha must all be positive")

    num_samples_scaled = round(x.size * (f_target / f_orig) / alpha)

    if num_samples_scaled < 2:
        raise ValueError(
            f"alpha={alpha} is too large for an input of {x.size} samples "
            f"(would produce {num_samples_scaled} output sample(s))"
        )
    if num_samples_scaled > max_output_samples:
        raise ValueError(
            f"Computed output length {num_samples_scaled} exceeds "
            f"max_output_samples={max_output_samples}. Check f_orig/f_target/"
            f"alpha for a unit mismatch, or raise max_output_samples if this "
            f"clip length is intentional."
        )

    return signal.resample(x, num_samples_scaled)


def normalize_peak(x: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """
    Normalizes the peaks of a signal to [-target_peak, target_peak].

    :param x: Input signal.
    :param target_peak: Maximum absolute peak level (0.0 < target_peak <= 1.0).
    :return: The normalized audio.
    """
    if not (0.0 < target_peak <= 1.0):
        raise ValueError("target_peak must be between 0 and 1")

    out = np.asarray(x, dtype=np.float64).copy()
    max_val = np.max(np.abs(out))

    if max_val > 0:
        out = (out / max_val) * target_peak
    return out


def sonify_signal(
        x: np.ndarray,
        f_orig: float,
        f_target: int = 8000,
        alpha: float = 40000,
        max_output_samples: int = MAX_OUTPUT_SAMPLES,
        target_peak: float = 0.95,
) -> np.ndarray:
    """
    Resamples, time-compresses, and peak-normalizes a filtered velocity
    signal into an audible waveform.

    Parameters should be chosen so that alpha * f_orig lands within the
    audible range [20 Hz, 20000 Hz] -- e.g. for HMI's ~3.3 mHz p-mode band,
    alpha=40000 maps to roughly 130 Hz.

    :param x: Filtered, gap-free velocity signal (e.g. output of
        filter_segments + zero_fill_with_taper). Must contain no NaN/Inf.
    :param f_orig: Sampling frequency (Hz) of x -- i.e. 1 / actual cadence
        used to produce x. No default: must match the real data cadence.
    :param f_target: Output audio sample rate (Hz).
    :param alpha: Time-compression factor mapping infrasonic solar
        frequencies into the audible range.
    :param max_output_samples: Passed through to resample_and_compress; see
        its docstring.
    :param target_peak: Peaks to normalize to. Should be between 0 and 1.
    :return: Normalized audio waveform, peak-scaled to [-target_peak, target-Peak],
        sampled at f_target Hz.
    """
    compressed = resample_and_compress(
        x,
        f_orig=f_orig,
        f_target=f_target,
        alpha=alpha,
        max_output_samples=max_output_samples,
    )
    return normalize_peak(compressed, target_peak=target_peak)


def audio_timeline(audio: np.ndarray, fs: float) -> np.ndarray:
    """
    Playback timestamps (seconds) for a uniformly-sampled audio array at `fs` Hz.
    Used to plot playback time and audio signal.
    """
    return np.arange(len(audio), dtype=np.float64) / fs