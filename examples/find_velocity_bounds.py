"""
Calculates basic statistical values for a raw and
a bandpass-filtered Doppler velocity signal, particularly
the min, max, 1st and 99th percentiles, mean, and standard deviation.

Helpful for calculating v_min and v_max for generating image colormaps.

The script calculates these values by getting the bounds of
a velocity value of one position on the solar disk -- which
should roughly estimate the endpoint values of the entire disk.

By default, this script uses the cached dataset for
2020-01-01 to 2020-01-08 at the center of the solar disk
(x=256, y=256). If the required FITS files are not present,
you will be prompted to download them.
"""
import numpy as np
from astropy.time import Time

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

    # Load velocity time series
    times, velocities = get_velocity_timeseries(
        start_time=start_time,
        end_time=end_time,
        x=x,
        y=y,
        timeseries_data_dir=EXAMPLE_DATA_DIR,
    )

    # Raw velocity
    v_min, v_max = np.nanmin(velocities), np.nanmax(velocities)
    v_1, v_99 = np.nanpercentile(velocities, 1), np.nanpercentile(velocities, 99)
    v_mean, v_std = np.nanmean(velocities), np.nanstd(velocities)
    print("-- Raw Velocity --\n"
          f"Minimum: {v_min:.2f} m/s\n"
          f"Maximum: {v_max:.2f} m/s\n"
          f"1st percentile: {v_1:.2f} m/s\n"
          f"99th percentile: {v_99:.2f} m/s\n"
          f"Mean: {v_mean:.2f} m/s\n"
          f"Standard deviation: {v_std:.2f} m/s\n")

    # Filtered velocity
    velocities = interpolate_short_gaps(velocities, max_gap=3)
    velocities = filter_segments(velocities, low_cutoff=0.001, high_cutoff=0.005, filter_order=2)
    velocities = zero_fill_with_taper(velocities)

    v_min, v_max = np.nanmin(velocities), np.nanmax(velocities)
    v_1, v_99 = np.nanpercentile(velocities, 1), np.nanpercentile(velocities, 99)
    v_mean, v_std = np.nanmean(velocities), np.nanstd(velocities)
    print("-- Filtered Velocity --\n"
          f"Minimum: {v_min:.2f} m/s\n"
          f"Maximum: {v_max:.2f} m/s\n"
          f"1st percentile: {v_1:.2f} m/s\n"
          f"99th percentile: {v_99:.2f} m/s\n"
          f"Mean: {v_mean:.2f} m/s\n"
          f"Standard deviation: {v_std:.2f} m/s\n")


if __name__ == "__main__":
    main()