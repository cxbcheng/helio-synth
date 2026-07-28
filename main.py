from pathlib import Path

from astropy.visualization import time_support
from matplotlib import pyplot as plt

from src.data_ingest.client import download_dopplergram_fits
from src.data_ingest.extraction import extract_velocity_timeseries
from src.data_ingest.storage import load_timeseries_npz, save_timeseries_npz
from src.data_ingest.utils import clean_drms_timestamp


def main():
    print('Initializing HelioSynth...')
    times, velocities = run_ingest()

    # Example plot
    time_support(format='iso', simplify=True)
    p = plt.plot(times, velocities)
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title('Velocity vs Time at the Center of the Solar Disk')
    plt.grid(True)
    plt.show()
    # TODO: Load UI and initialize SunPy data pipeline

def run_ingest():
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'

    # ~4 hours download time
    start_time = "2020.01.01_00:00:00_TAI"
    end_time = "2020.01.08_00:00:00_TAI"
    scale = 0.125
    cadence = 45

    # Spatial coordinates to gather velocity timeseries v(x, y, t)
    x = 256
    y = 256

    # Get directory for the particular data given
    query_folder_name = clean_drms_timestamp(f"hmi_v_45s_{start_time}_{end_time}_{scale}_{cadence}")
    raw_cache_dir = Path(raw_data_dir) / query_folder_name
    existing_fits = list(raw_cache_dir.glob("*.fits")) if raw_cache_dir.exists() else []

    if not existing_fits:
        d = input(f"Cannot find existing FITS files for the directory {raw_cache_dir.absolute()}.\nDownload? (y/n)")
        if d.lower() == 'y':
            download_dopplergram_fits(
                start_time=start_time,
                end_time=end_time,
                scale=scale,
                cadence=cadence,
                download_dir=raw_cache_dir,
            )
        else:
            # TODO: further handling
            raise SystemExit()
    else:
        print(f"Found {len(existing_fits)} FITS files for the directory {raw_cache_dir}.")

    # Check processed cache to see if times, velocities NumPy arrays have already been stored
    processed_cache_file = Path(processed_data_dir) / f"{query_folder_name}_x{x}_y{y}.npz"

    if processed_cache_file.exists():
        print(f"Getting velocity timeseries from {processed_cache_file}")
        times, velocities = load_timeseries_npz(path=processed_cache_file)
    else:
        print(f"Extracting velocity timeseries...")
        times, velocities = extract_velocity_timeseries(data_dir=raw_cache_dir, x=x, y=y, cadence=cadence, start_index=0, end_index=1000)
        print(f"Saving velocity timeseries to {processed_cache_file}")
        save_timeseries_npz(path=processed_cache_file, times=times, velocities=velocities, cadence=cadence)

    return times, velocities


if __name__ == '__main__':
    main()