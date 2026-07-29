"""
Plots a raw Doppler velocity time series from HMI observations.

By default, this script uses the cached dataset for
2020-01-01 to 2020-01-08 at the center of the solar disk
(x=256, y=256). If the required FITS files are not present,
you will be prompted to download them.
"""

from matplotlib import pyplot as plt

from heliosynth.data_ingest.ingest import run_ingest


def main():
    # Change parameters here to test different regions/timeframes
    start_time = "2020.01.01_00:00:00_TAI"
    end_time = "2020.01.08_00:00:00_TAI"
    x = 256
    y = 256

    # Load velocity time series
    times, velocities = run_ingest(start_time=start_time, end_time=end_time, x=x, y=y)
    elapsed = (times - times[0]).sec

    plt.figure()
    plt.plot(elapsed, velocities)
    plt.xlabel('Time since 2020-01-01 00:00:00 [s]')
    plt.ylabel('Doppler Velocity [m/s]')
    plt.title('Velocity vs Time at the Center of the Solar Disk')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()