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
spacecraft orbital trend (~24-hour period, ≈0.012 mHz) while isolating
the Sun's 5-minute p-mode oscillations (~3 mHz). Narrower passbands
emphasize oscillations near the selected frequencies.
"""
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from scipy.fft import rfft, rfftfreq

from heliosynth.data_ingest.ingest import run_ingest
from heliosynth.paths import EXAMPLE_DATA_DIR, AUDIO_DATA_DIR
from heliosynth.processing.cleaning import interpolate_short_gaps, zero_fill_with_taper
from heliosynth.processing.filter import filter_segments
from heliosynth.processing.sonification import sonify_signal, audio_timeline


def main():
    # Change parameters here to test different regions/timeframes
    start_time = "2020.01.01_00:00:00_TAI"
    end_time = "2020.01.08_00:00:00_TAI"
    x = 256
    y = 256
    # Bandpass filter parameters; 5-minute oscillations = ~3 mHz
    low_cutoff = 0.001  # 1 mHz
    high_cutoff = 0.005  # 5 mHz

    # To make the audio audible; increase to speed + pitch up
    # In general, the new audio will be in the frequency range:
    # [low_cutoff, high_cutoff] * alpha * sample_rate
    alpha = 80000

    # Audio parameters
    sample_rate = 8000
    audio_path = AUDIO_DATA_DIR / 'solar_velocity.wav'

    # Load velocity time series
    times, velocities = run_ingest(start_time=start_time, end_time=end_time, x=x, y=y,
                                   processed_data_dir=EXAMPLE_DATA_DIR)
    elapsed = (times - times[0]).sec

    velocities = interpolate_short_gaps(velocities, max_gap=3)
    velocities = filter_segments(velocities, low_cutoff=low_cutoff, high_cutoff=high_cutoff, filter_order=2)
    velocities = zero_fill_with_taper(velocities)
    audio = sonify_signal(velocities, f_orig=1/45, f_target=sample_rate, alpha=alpha)

    print(f"Downloading signal audio to {audio_path}...")
    sf.write(AUDIO_DATA_DIR / 'solar_audio.wav', audio, sample_rate)

    times = audio_timeline(audio, sample_rate)

    # Compute FFT of the sonified audio signal
    n_samples = len(audio)
    yf = np.abs(rfft(audio))
    xf = rfftfreq(n_samples, 1 / sample_rate)

    # Waveform + frequency spectrum plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Waveform plot
    ax1.plot(audio, color='steelblue', lw=0.5)
    ax1.set_xlabel('Audio Playback Time')
    ax1.set_ylabel('Audio Amplitude')
    ax1.set_title(f"Sonified Audio Waveform "
              f"({alpha * sample_rate}x Playback Speed)")
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