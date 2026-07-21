import numpy as np
from scipy import signal
import soundfile as sf

def sonify_solar(v: np.ndarray,
                 dt: float,
                 f_target: int,
                 alpha: float,
                 low_cutoff: float = 0.0010,
                 high_cutoff: float = 0.0050) -> np.ndarray:
    """
    TODO: REWRITE DOCUMENTATION TO BE MORE DESCRIPTIVE OF THE PROCESS
    TODO: DOCUMENT RATIONALE BEHIND ~3mHz +/- 2 mHz CUTOFFS
    Applies a Butterworth bandpass filter and an alpha time-scaling operator to translate
    an infrasonic velocity time-series array into an audible waveform.
    :param v: 1D time-series vector of velocity measurements v[k] representing
        Doppler velocity measurements v(t) at a given photosphere coordinate.
        Unit: m/s
    :param dt: Sampling period between two consecutive measurements v[k] and v[k+1],
        i.e. the inverse of the original sampling frequency. Unit: s
    :param f_target: The target sampling rate of the output audio. Unit: Hz
    :param alpha: Time-scaling compression factor. Since solar p-modes are inaudible,
        use a large alpha value such that alpha*f_solar \in [20, 20000] Hz,
        i.e. the human hearing range.
    :param low_cutoff: Low cutoff frequency for bandpass filter.
    :param high_cutoff: High cutoff frequency for bandpass filter.
    :return: 1D audio waveform array bounded to [-1.0, 1.0].
    """
    # Original sampling frequency
    f_orig = 1.0 / dt

    # TODO: REFACTOR BANDPASS FILTER, TIME COMPRESS, AND NORMALIZE PEAK INTO SEPARATE FUNCTIONS

    # Butterworth bandpass filter
    # TODO: ADD MORE PARAMETERS, E.G. FILTER_ORDER
    sos = signal.butter(N=4,
                        Wn=[low_cutoff, high_cutoff],
                        fs=f_orig,
                        btype='bandpass',
                        output='sos')
    v_filtered = signal.sosfiltfilt(sos, v)

    # Resample to target sampling rate
    num_samples_target = v.size * (f_target / f_orig)

    # Time-scale
    # TODO: CONSIDER ROUNDING INSTEAD OF FLOORING
    num_samples_scaled = int(num_samples_target / alpha)
    v_audio = signal.resample(v_filtered, num_samples_scaled)

    # Normalize max amplitude to 1.0
    max_val = np.max(np.abs(v_audio))
    if max_val > 0:
        v_audio /= max_val

    # TODO: REMOVE WRITE TO FILE WHEN DONE
    sf.write('solar_oscillation.wav', v_audio, f_target)
    return v_audio

# TODO: REMOVE TESTS
dt = 1/44100 # sampling interval
duration = 2.0 # seconds

t = np.arange(0, duration, dt)
v = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*277*t) + np.sin(2*np.pi*330*t)
audio = sonify_solar(v, dt, f_target=88200, alpha=1)
# print(audio.size)