"""
Plots a bandpass-filtered HMI Doppler velocity time series and generates
a sonified version of the filtered signal.

This script demonstrates the complete sonification pipeline:

    HMI Doppler velocities
        - interpolate short gaps
        - bandpass filter
        - taper missing segments
        - time-compress and pitch-shift
        - save as WAV
        - visualize waveform and frequency spectrum

The default bandpass of [1, 5] mHz removes the dominant long-period
spacecraft orbital trend (~24-hour period = 0.012 mHz) while isolating
the Sun's 5-minute p-mode oscillations (~3 mHz). Narrower passbands
emphasize oscillations near the selected frequencies.

For more information on the bandpass filter, see
scripts/plot_signal_bandpass.py.
"""
import numpy as np
from astropy.time import Time
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq

from heliosynth.data_ingest.ingest import get_velocity_timeseries
from heliosynth.paths import EXAMPLE_DATA_DIR, EXAMPLE_DOWNLOAD_DIR
from heliosynth.processing.cleaning import interpolate_short_gaps, zero_fill_with_taper
from heliosynth.processing.filter import filter_segments
from heliosynth.processing.sonification import sonify_signal, audio_timeline


def main():
    # Change parameters here to test different regions/timeframes
    start_time = Time('2020-01-01 00:00:00', scale='tai')
    end_time = Time('2020-01-08 00:00:00', scale='tai')
    x = 256
    y = 256
    # Bandpass filter parameters; 5-minute oscillations = ~3 mHz
    low_cutoff = 0.001  # 1 mHz
    high_cutoff = 0.005  # 5 mHz

    """
    alpha is the playback speed of the samples. An alpha of 1 is real time;
    an alpha of 86400 plays 1 day of audio per 1 second.
    
    Make the audio audible by increasing this value.
    In general, the new audio will be in the frequency range:
    
    [low_cutoff, high_cutoff] * alpha * sample_rate
    """
    alpha = 80000

    # Audio parameters
    sample_rate = 8000
    audio_path = EXAMPLE_DOWNLOAD_DIR / 'solar_audio.wav'

    # Load velocity time series
    times, velocities = get_velocity_timeseries(
        start_time=start_time,
        end_time=end_time,
        x=x,
        y=y,
        timeseries_data_dir=EXAMPLE_DATA_DIR)

    velocities = interpolate_short_gaps(velocities, max_gap=3)
    velocities = filter_segments(velocities, low_cutoff=low_cutoff, high_cutoff=high_cutoff, filter_order=2)
    velocities = zero_fill_with_taper(velocities)
    audio = sonify_signal(velocities, f_orig=1/45, f_target=sample_rate, alpha=alpha)

    print(f"Downloading signal audio to {audio_path}...")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(audio_path, sample_rate, audio.astype(np.float32))

    # Compute FFT of the sonified audio signal
    n_samples = len(audio)
    yf = np.abs(rfft(audio))
    xf = rfftfreq(n_samples, 1 / sample_rate)

    # Waveform + frequency spectrum plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Waveform plot
    ax1.plot(audio_timeline(audio, fs=sample_rate), audio, color='steelblue', lw=0.5)
    ax1.set_xlabel('Audio Playback Time [s]')
    ax1.set_ylabel('Audio Amplitude')
    ax1.set_title(f"Sonified Audio Waveform ({alpha}x Playback Speed)")
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True)

    # Frequency spectrum plot
    ax2.plot(xf, yf, color='darkorange', lw=1.2)
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Spectral Magnitude')
    ax2.set_title('Frequency Content (Solar p-modes pitch-shifted into audible range)')
    ax2.set_xlim(20, 2000)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()