"""
Handles processing and caching of fits files for quick random access
in the main app.
"""
import logging

import numpy as np
from astropy.io import fits

from heliosynth.data_ingest.extraction import get_fits_files
from heliosynth.data_ingest.utils import get_dataset_dir, fits_name_to_time
from heliosynth.paths import RAW_DATA_DIR
from heliosynth.sampling.vogel import construct_vogel_spiral

logger = logging.getLogger(__name__)


def main():
    scale = 0.125
    cadence = 45
    im_width = round(4096 * scale)

    fits_files = get_fits_files(get_dataset_dir(RAW_DATA_DIR, scale, cadence))
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

    # We will treat velocity as a function of sample points over time.
    # Specifically, we express the velocity series as an M x N matrix
    # where M = n_timestamps and N = n_points.
    # For convenience, we will list timestamps and points as indices.
    velocity_series = np.full((n_timestamps, n_points), np.nan)
    frames = np.full((n_timestamps, n_points), np.nan)

    for t, fits_file in enumerate(fits_files[:5]):
        with fits.open(fits_file) as hdul:
            # Dopplergram image
            im = hdul[1].data

            # Store velocity signals
            for i, (row, col) in enumerate(sample_pixels):
                velocity = im[row, col]
                velocity_series[t][i] = velocity

            # Store images
            # TODO



    print(velocity_series)


if __name__ == "__main__":
    main()