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

from heliosynth.constants import V_MIN, V_MAX
from heliosynth.data_ingest.extraction import get_fits_files, extract_solar_image
from heliosynth.data_ingest.storage import save_disk_velocity_zarr
from heliosynth.path_utils import get_dataset_dir, fits_name_to_time, disk_velocity_zarr_path, doppler_image_path
from heliosynth.paths import RAW_DATA_DIR, DATASETS_DATA_DIR
from heliosynth.sampling.vogel import construct_vogel_spiral

logger = logging.getLogger(__name__)


def main():
    # Processing parameters
    scale = 0.125
    res = round(scale * 4096)
    cadence = 45
    im_width = round(4096 * scale)
    start_time = Time('2020-01-01 00:00:00', scale='tai')
    end_time = Time('2020-01-01 00:10:00', scale='tai')
    n_points = 4000
    image_options = [('RdBu_r', 0), ('RdBu_r', 2), ('plasma', 0), ('plasma', 2)]

    raw_dataset_dir = get_dataset_dir(RAW_DATA_DIR, res, cadence)
    dataset_dir = get_dataset_dir(DATASETS_DATA_DIR, res, cadence)
    fits_files = get_fits_files(raw_dataset_dir, start_time, end_time)

    sample_points = construct_vogel_spiral(
        n_points=n_points,
        radius=im_width / 2,
        snap_to_nearest_integer=True,
        include_center=True
    )

    # (x, y) centered coordinate system -> (row, col) image coordinate system
    sample_pixels = np.column_stack((
        im_width // 2 - sample_points[:, 1],  # row
        im_width // 2 + sample_points[:, 0],  # col
    ))
    sample_pixels = np.clip(sample_pixels, 0, im_width - 1)
    rows, cols = sample_pixels[:, 0], sample_pixels[:, 1]

    velocity_rows, t_recs = [], []

    for t, fits_file in enumerate(fits_files):
        try:
            with fits.open(fits_file) as hdul:
                # Dopplergram image
                im = hdul[1].data
        except (OSError, IOError, IndexError) as e:
            logger.warning("Skipping %s: %s", fits_file.name, e)
            continue

        t_rec = fits_name_to_time(fits_file.name)
        t_recs.append(t_rec)
        velocity_rows.append(im[rows, cols])

        for cmap, detrend_order in image_options:
            rendered = extract_solar_image(im, out_size=im_width,
                v_min=V_MIN[detrend_order],
                v_max=V_MAX[detrend_order],
                colormap=cmap, detrend_order=detrend_order)
            im_path = doppler_image_path(dataset_dir, t_rec, detrend_order, cmap, 'webp')
            im_path.parent.mkdir(parents=True, exist_ok=True)
            rendered.save(im_path, format='WEBP')

    if not velocity_rows:
        raise ValueError(f"No valid samples found in {fits_files!r}")

    velocity_series = np.array(velocity_rows, dtype=np.float32)
    times = Time(t_recs, scale='tai')

    zarr_path = disk_velocity_zarr_path(dataset_dir, n_points)
    save_disk_velocity_zarr(zarr_path, times, velocity_series, sample_points, im_width)
    logger.debug("Saved %d frames x %d points to %s", *velocity_series.shape, zarr_path)


if __name__ == "__main__":
    main()
