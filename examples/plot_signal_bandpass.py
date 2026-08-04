"""
Plots a Doppler velocity time series from HMI observations
as a signal with a bandpass filter applied.
The default filter settings are [1 mHz, 5 mHz] in order
to (1) detrend the satellite orbit velocity (which orbits
once every 24 hours, so an oscillation of 0.012 mHz)
and to isolate the Sun's 5-minute oscillations (~3 mHz).
Filtering closer to this frequency will make this frequency more prevalent.

By default, this script uses the cached dataset for
2020-01-01 to 2020-01-08 at the center of the solar disk
(x=256, y=256). If the required FITS files are not present,
you will be prompted to download them.
"""
from astropy.time import Time
from matplotlib import pyplot as plt

from heliosynth.data_ingest.ingest import get_velocity_timeseries
from heliosynth.paths import EXAMPLE_DATA_DIR
from heliosynth.processing.cleaning import interpolate_short_gaps, zero_fill_with_taper
from heliosynth.processing.filter import filter_segments


def main():
    # Change parameters here to test different regions/timeframes
    start_time = Time('2020-01-01 00:00:00', scale='tai')
    end_time = Time('2020-01-08 00:00:00', scale='tai')
    x = 256
    y = 256
    # Bandpass filter parameters; 5-minute oscillations = ~3 mHz
    low_cutoff = 0.001  # 1 mHz
    high_cutoff = 0.005  # 5 mHz

    # Matplotlib plot parameters
    plt_x_start = 0
    plt_x_end = plt_x_start + 3600  # 60 minute duration

    # Load velocity time series
    times, velocities = get_velocity_timeseries(
        start_time=start_time,
        end_time=end_time,
        x=x,
        y=y,
        timeseries_data_dir=EXAMPLE_DATA_DIR)
    elapsed = (times - times[0]).sec

    velocities = interpolate_short_gaps(velocities, max_gap=3)
    velocities = filter_segments(velocities, low_cutoff=low_cutoff, high_cutoff=high_cutoff, filter_order=2)
    velocities = zero_fill_with_taper(velocities)

    plt.figure()
    plt.plot(elapsed, velocities)
    plt.xlabel(f'Time since {start_time.iso} [s]')
    plt.ylabel('Doppler Velocity [m/s]')
    plt.title('Filtered Signal at the Center of the Solar Disk')
    plt.xlim(plt_x_start, plt_x_end)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()