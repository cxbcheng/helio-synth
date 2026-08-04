"""
Handles processing and caching of FITS files for quick random access
in the main app.

Only processes the FITS files from the raw data directory.
This should be executed after downloading new data.
"""
import logging

import numpy as np
from astropy.io import fits
from astropy.time import Time

from heliosynth.data_ingest.extraction import get_fits_files, extract_solar_image
from heliosynth.data_ingest.utils import get_dataset_dir, fits_name_to_time
from heliosynth.paths import RAW_DATA_DIR, IMAGES_DATA_DIR
from heliosynth.sampling.vogel import construct_vogel_spiral

logger = logging.getLogger(__name__)


def main():
    scale = 0.125
    cadence = 45
    im_width = round(4096 * scale)
    start_time = Time('2020-01-01 00:00:00', scale='tai')
    end_time = Time('2020-01-02 00:00:00', scale='tai')

    im_dir = get_dataset_dir(IMAGES_DATA_DIR, scale, cadence)
    im_dir.mkdir(parents=True, exist_ok=True)
    fits_files = get_fits_files(get_dataset_dir(RAW_DATA_DIR, scale, cadence), start_time, end_time)

    n_timestamps = len(fits_files)
    n_points = 2000

    sample_points = construct_vogel_spiral(
        n_points=n_points,
        radius=im_width/2,
        snap_to_nearest_integer=True,
        include_center=True
    )

    # (x, y) centered coordinate system -> (row, col) image coordinate system
    sample_pixels = np.column_stack((
        im_width // 2 - sample_points[:, 1],  # row
        im_width // 2 + sample_points[:, 0],  # col
    ))
    sample_pixels = np.clip(sample_pixels, 0, im_width - 1)

    # We will treat velocity as a function of sample points over time.
    # Specifically, we express the velocity series as an M x N matrix
    # where M = n_timestamps and N = n_points.
    # For convenience, we will list timestamps and points as indices.
    velocity_series = np.full((n_timestamps, n_points), np.nan)

    for t, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            # Dopplergram image
            im = hdul[1].data

            # Store velocity signals
            for i, (row, col) in enumerate(sample_pixels):
                velocity = im[row, col]
                velocity_series[t][i] = velocity

            # Store image
            rendered = extract_solar_image(im, out_size=im_width, detrend_order=0)
            im_filename = fits_file.stem + '.webp'
            rendered.save(im_dir / im_filename, format='WEBP')


    # TODO: save locally
    logger.debug(velocity_series)


if __name__ == "__main__":
    main()