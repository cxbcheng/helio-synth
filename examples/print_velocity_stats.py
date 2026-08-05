"""
Calculates basic statistical values for a raw and a bandpass-filtered
Doppler velocity signal and prints to console.

Helpful for calculating v_min and v_max needed for generating image colormaps.
The 99th percentile absolute velocity is a good upper bound and negative lower
bound while eliminating significant outliers that may be present.

The script calculates these values by getting the bounds of
a velocity value of one position on the solar disk -- which
should roughly estimate the endpoint values of the entire disk.

A helper method is provided to print the relevant stats.

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


def print_velocity_stats(velocities, label):
    abs_velocities = np.abs(velocities)

    stats = {
        "min": np.nanmin(velocities),
        "max": np.nanmax(velocities),
        "p1": np.nanpercentile(velocities, 1),
        "p99": np.nanpercentile(velocities, 99),
        "abs_p1": np.nanpercentile(abs_velocities, 1),
        "abs_p99": np.nanpercentile(abs_velocities, 99),
        "mean": np.nanmean(velocities),
        "std": np.nanstd(velocities),
    }

    print(
        f"------------------------------------------------------------\n"
        f"{label} Velocity\n"
        f"------------------------------------------------------------\n"
        f"Minimum: {stats['min']:.2f} m/s\n"
        f"Maximum: {stats['max']:.2f} m/s\n"
        f"1st percentile: {stats['p1']:.2f} m/s\n"
        f"99th percentile: {stats['p99']:.2f} m/s\n"
        f"Absolute Velocity 1st percentile: {stats['abs_p1']:.2f} m/s\n"
        f"Absolute Velocity 99th percentile: {stats['abs_p99']:.2f} m/s\n"
        f"Mean: {stats['mean']:.2f} m/s\n"
        f"Standard deviation: {stats['std']:.2f} m/s\n"
    )


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
    print_velocity_stats(velocities, "Raw")

    # Filtered velocity
    velocities = interpolate_short_gaps(velocities, max_gap=3)
    velocities = filter_segments(velocities, low_cutoff=0.001, high_cutoff=0.005, filter_order=2)
    velocities = zero_fill_with_taper(velocities)
    print_velocity_stats(velocities, "Filtered")


if __name__ == "__main__":
    main()