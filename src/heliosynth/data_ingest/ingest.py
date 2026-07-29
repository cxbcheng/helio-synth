from pathlib import Path

import numpy as np
from astropy.time import Time

from src.heliosynth.data_ingest.client import download_dopplergram_fits
from src.heliosynth.data_ingest.extraction import extract_velocity_timeseries
from src.heliosynth.data_ingest.storage import load_timeseries_npz, save_timeseries_npz
from src.heliosynth.data_ingest.utils import make_query_name
from src.heliosynth.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


def run_ingest(
        start_time: str = "2020.01.01_00:00:00_TAI",
        end_time: str = "2020.01.08_00:00:00_TAI",
        scale: float = 0.125,
        cadence: int = 45,
        x: int = 256,
        y: int = 256,
        raw_data_dir: str | Path = RAW_DATA_DIR,
        processed_data_dir: str | Path = PROCESSED_DATA_DIR,
        prompt_download: bool = True,
) -> tuple[Time, np.ndarray]:
    """
    Downloads (if necessary), extracts, caches, and returns a Doppler velocity
    time series for a specified spatial location and time range.
    Note that the time format should fit "2020.01.01_00:00:00_TAI", i.e. a period to
    delimit the date, an underscore between the date and time, and a suffix _TAI.
    This directly matches the DRMS query time parameter format, and it will also be used
    (although in a cleaner format determined by `make_query_name`) to look up from or
    download into the raw data folder.

    :param start_time: Start of the DRMS query time range.
    :param end_time: End of the DRMS query time range.
    :param scale: Spatial downsampling factor used when downloading FITS files.
    :param cadence: Temporal cadence of the Dopplergrams, in seconds.
    :param x: Horizontal pixel coordinate of the extracted time series.
    :param y: Vertical pixel coordinate of the extracted time series.
    :param raw_data_dir: Directory containing cached raw FITS files.
    :param processed_data_dir: Directory containing cached processed time series.
    :param prompt_download: Whether to prompt the user before downloading.
    :return: (times, velocities). Velocity unit: m/s
    """
    # Get directory for the particular data given
    query_folder_name = make_query_name(start_time=start_time, end_time=end_time, scale=scale, cadence=cadence)
    raw_cache_dir = raw_data_dir / query_folder_name
    existing_fits = list(raw_cache_dir.glob("*.fits")) if raw_cache_dir.exists() else []

    if not existing_fits:
        if prompt_download:
            res = input(f"Cannot find existing FITS files for the directory {raw_cache_dir.absolute()}.\nDownload? (y/n)")
            if res.lower() != 'y':
                print("Exiting...")
                raise SystemExit()

        download_dopplergram_fits(
            start_time=start_time,
            end_time=end_time,
            scale=scale,
            cadence=cadence,
            download_dir=raw_cache_dir,
        )
    else:
        print(f"Found {len(existing_fits)} FITS files for the directory {raw_cache_dir}.")

    # Check processed cache to see if times, velocities NumPy arrays have already been stored
    processed_cache_file = processed_data_dir / f"{query_folder_name}_x{x}_y{y}.npz"

    if processed_cache_file.exists():
        print(f"Getting velocity timeseries from {processed_cache_file}")
        times, velocities = load_timeseries_npz(path=processed_cache_file)
    else:
        print(f"Extracting velocity timeseries...")
        times, velocities = extract_velocity_timeseries(data_dir=raw_cache_dir, x=x, y=y, cadence=cadence)
        print(f"Saving velocity timeseries to {processed_cache_file}")
        save_timeseries_npz(path=processed_cache_file, times=times, velocities=velocities, cadence=cadence)

    return times, velocities