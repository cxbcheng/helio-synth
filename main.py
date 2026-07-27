from pathlib import Path

from src.data_ingest.client import download_dopplergram_fits
from src.data_ingest.extraction import extract_velocity_timeseries
from src.data_ingest.utils import clean_drms_timestamp


def main():
    print('Initializing HelioSynth...')
    run_ingest()
    # TODO: Load UI and initialize SunPy data pipeline

def run_ingest():
    raw_data_dir = 'data/raw'
    start_time = "2020.01.01_00:00:00_TAI"
    end_time = "2020.01.08_00:00:00_TAI"
    scale = 0.125
    cadence = 45

    # Get directory for the particular data given
    query_folder_name = clean_drms_timestamp(f"hmi_v_45s_{start_time}_{end_time}_{scale}_{cadence}")
    cache_dir = Path(raw_data_dir) / query_folder_name
    existing_fits = list(cache_dir.glob("*.fits")) if cache_dir.exists() else []

    if not existing_fits:
        d = input(f"Cannot find existing FITS files for the directory {cache_dir.absolute()}.\nDownload? (y/n)")
        if d.lower() == 'y':
            download_dopplergram_fits(
                start_time=start_time,
                end_time=end_time,
                scale=scale,
                cadence=cadence,
                download_dir=cache_dir,
            )
        else:
            # TODO: further handling
            raise SystemExit()
    else:
        print(f"Found {len(existing_fits)} FITS files for the directory {cache_dir.absolute()}.")

    velocity_data = extract_velocity_timeseries(data_dir=cache_dir, x=0, y=0)
    print(f"Extraction complete.")

if __name__ == '__main__':
    main()