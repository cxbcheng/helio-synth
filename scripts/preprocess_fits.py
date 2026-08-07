"""
Handles processing and caching of FITS files for quick random access
in the main app.

Only processes the FITS files from the raw data directory.
This should be executed after downloading new data.
"""
import json
import logging

import numpy as np
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

from heliosynth.constants import V_MIN, V_MAX, DEFAULT_DISK_RADIUS_FRACTION
from heliosynth.data_ingest.extraction import get_fits_files, get_disk_mask
from heliosynth.data_ingest.storage import save_disk_velocity_zarr
from heliosynth.path_utils import get_dataset_dir, fits_name_to_time, disk_velocity_zarr_path, doppler_image_path
from heliosynth.paths import RAW_DATA_DIR, DATASETS_DATA_DIR
from heliosynth.processing.imaging import detrend_disk_surface, render_dopplergram
from heliosynth.sampling.vogel import construct_vogel_spiral

logger = logging.getLogger(__name__)


def main():
    # Processing parameters
    res = 512
    cadence = 45
    start_time = Time('2020-01-01 00:00:00', scale='tai')
    end_time = Time('2020-02-01 00:00:00', scale='tai')
    n_points = 4000
    colormaps = ['RdBu_r', 'plasma']
    detrend_orders = [0, 2]
    time_chunk = 1024

    raw_dataset_dir = get_dataset_dir(RAW_DATA_DIR, res, cadence)
    dataset_dir = get_dataset_dir(DATASETS_DATA_DIR, res, cadence)
    fits_files = get_fits_files(raw_dataset_dir, start_time, end_time)
    solar_radius = res / 2

    sample_points = construct_vogel_spiral(
        n_points=n_points,
        radius=solar_radius * DEFAULT_DISK_RADIUS_FRACTION,
        snap_to_nearest_integer=True,
        include_center=True
    )

    # (x, y) centered coordinate system -> (row, col) image coordinate system
    sample_pixels = np.column_stack((
        res // 2 - sample_points[:, 1],  # row
        res // 2 + sample_points[:, 0],  # col
    ))
    sample_pixels = np.clip(sample_pixels, 0, res - 1)
    rows, cols = sample_pixels[:, 0], sample_pixels[:, 1]

    velocity_rows, t_recs = [], []

    for t, fits_file in enumerate(tqdm(fits_files, desc='Processing FITS files', unit='files')):
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

        for detrend_order in detrend_orders:
            disk_mask = get_disk_mask(im)
            im_detrended = detrend_disk_surface(im, disk_mask, detrend_order)

            for cmap in colormaps:
                rendered = render_dopplergram(im_detrended, disk_mask,
                    V_MIN[detrend_order], V_MAX[detrend_order],
                    out_size=res, colormap=cmap)

                im_path = doppler_image_path(dataset_dir, t_rec, detrend_order, cmap, 'webp')
                im_path.parent.mkdir(parents=True, exist_ok=True)
                rendered.save(im_path, format='WEBP')

    if not velocity_rows:
        raise ValueError(f"No valid samples found in {fits_files!r}")

    velocity_series = np.array(velocity_rows, dtype=np.float32)
    times = Time(t_recs, scale='tai')

    zarr_path = disk_velocity_zarr_path(dataset_dir, n_points)
    save_disk_velocity_zarr(zarr_path, times, velocity_series, sample_points, res, time_chunk)
    logger.debug("Saved %d frames x %d points to %s", *velocity_series.shape, zarr_path)

    # Save metadata
    data = {
        "width": res,
        "height": res,
        "cadence": cadence,
        "images": {
            "colormaps": colormaps,
            "detrend_orders": detrend_orders,
        },
        "timeseries": {
            "samples": [n_points],
        },
    }
    with open(dataset_dir / 'metadata.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
