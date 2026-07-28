import numpy as np
from numpy import ndarray
from scipy import signal

def sonify_solar(v: ndarray,
                 dt: float,
                 alpha: float = 1,
                 fs: int = 44100,
                 low_cutoff: float = 0.0010,
                 high_cutoff: float = 0.0050) -> ndarray:
    """
    Sonifies HMI solar surface velocity data.
    Specifically, applies a Butterworth bandpass filter and an alpha time-scaling operator
    to translate an infrasonic velocity time-series array into an audible waveform.
    Parameters should be chosen so that the original signal, which is around ~3 mHz
    ("5-minute oscillation"), is compressed to fit within the human hearing range [0, 20000] Hz.
    The bandpass filter can be set to filter out the signal noise even further.
    :param v: The velocity time series v[k] representing Doppler velocity measurements v(t)
        at a given photosphere coordinate.
        Unit: m/s
    :param dt: The sampling interval between two consecutive measurements v[k] and v[k+1],
        i.e. the inverse of the original sampling frequency. Unit: s
    :param alpha: Time-scaling compression factor. Since solar p-modes are inaudible,
        use a large alpha value such that alpha*f_solar \in [20, 20000] Hz,
        i.e. the human hearing range.
    :param fs: The target sampling rate of the output audio. Unit: Hz
    :param low_cutoff: Low cutoff frequency for bandpass filter.
    :param high_cutoff: High cutoff frequency for bandpass filter.
    :return: The velocity time series sonified.
    """
    # Original sampling frequency
    f_orig = 1.0 / dt
    v_filtered = _butter_bandpass_filter(v, fs=f_orig, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
    return _normalize_peak(_resample_and_compress(v_filtered, f_orig, fs, alpha))


def _butter_bandpass_filter(x: ndarray, fs: float, low_cutoff: float, high_cutoff: float, filter_order=4) -> ndarray:
    """
    Applies a forward-backward Butterworth bandpass filter on a signal.
    :param x: The array of data to be filtered.
    :param fs: Sampling frequency of the signal.
    :param low_cutoff: Low cutoff frequency for bandpass filter.
    :param high_cutoff: High cutoff frequency for bandpass filter.
    :param filter_order: The order of the filter before forward-backward filtering.
        After forward-backward filtering is applied, the filter technically has double this order.
    :return: The filtered signal.
    """
    sos = signal.butter(N=filter_order,
                        Wn=[low_cutoff, high_cutoff],
                        fs=fs,
                        btype='bandpass',
                        output='sos')
    return signal.sosfiltfilt(sos, x)


def _resample_and_compress(x: ndarray, f_orig: float, f_target: int, alpha: float) -> ndarray:
    """
    Resamples an audio signal to a new sampling frequency, then compresses it by alpha.
    The resampling simply changes the size and precision of the audio signal, whereas
    compression changes a signal's perceived pitch.
    One may decide to only resample or only compress, but nonetheless these are grouped
    together so that only "signal.resample" is only performed once for both operations.
    :param x: The array of data to be normalized.
    :param alpha: Time compression scaling factor.
    :param f_orig: Sampling frequency of the original signal.
    :param f_target: Target sampling frequency of the compressed signal.
    :return: The resampled and compressed audio.
    """
    # Resample to target sampling rate
    num_samples_target = x.size * (f_target / f_orig)

    # Time-scale
    num_samples_scaled = round(num_samples_target / alpha)
    return signal.resample(x, num_samples_scaled)


def _normalize_peak(x: ndarray) -> ndarray:
    """
    Normalizes the peaks of a signal to [-1.0, 1.0].
    :param x: The array of data to be normalized.
    :return: The normalized audio.
    """
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x /= max_val
    return x