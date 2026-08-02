"""
Plots a raw Doppler velocity time series from HMI observations.

By default, this script uses the cached dataset for
2020-01-01 to 2020-01-08 at the center of the solar disk
(x=256, y=256). If the required FITS files are not present,
you will be prompted to download them.
"""
from astropy.time import Time
from matplotlib import pyplot as plt

from heliosynth.data_ingest.ingest import get_velocity_timeseries
from heliosynth.paths import EXAMPLE_DATA_DIR


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
    elapsed = (times - times[0]).sec

    plt.figure()
    plt.plot(elapsed, velocities)
    plt.xlabel(f'Time since {start_time} [s]')
    plt.ylabel('Doppler Velocity [m/s]')
    plt.title('Velocity vs Time at the Center of the Solar Disk')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()