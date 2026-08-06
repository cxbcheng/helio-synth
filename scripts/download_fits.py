"""
Downloads HMI Dopplergram FITS files from JSOC and stores them in RAW_DATA_DIR.

For more info, see `examples/download_hmi_data.py`.
"""
import logging

from astropy.time import Time

from heliosynth.data_ingest.client import download_dopplergram_fits
from heliosynth.path_utils import get_dataset_dir
from heliosynth.paths import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO)


def main():
    # Change parameters here to download different timeframes
    start_time = Time('2026-07-01 00:00:00', scale='tai')
    end_time = Time('2026-08-01 00:00:00', scale='tai')
    res = 1024
    scale = 4096 / res
    cadence = 45
    poll_interval = 300
    raw_dataset_dir = get_dataset_dir(RAW_DATA_DIR, res, cadence)

    download_dopplergram_fits(start_time,
                              end_time,
                              raw_data_dir=raw_dataset_dir,
                              scale=scale,
                              cadence=cadence,
                              poll_interval=poll_interval,
                              request_id=None)


if __name__ == "__main__":
    main()