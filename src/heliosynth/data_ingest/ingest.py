import logging
from pathlib import Path

import numpy as np
from astropy.time import Time

from heliosynth.data_ingest.client import download_dopplergram_fits
from heliosynth.data_ingest.extraction import extract_velocity_timeseries
from heliosynth.data_ingest.storage import load_timeseries_npz, save_timeseries_npz
from heliosynth.data_ingest.utils import make_query_name, get_dataset_dir_name
from heliosynth.paths import RAW_DATA_DIR, TIMESERIES_DATA_DIR
from heliosynth.time_utils import require_tai

logger = logging.getLogger(__name__)


def get_velocity_timeseries(
        start_time: Time,
        end_time: Time,
        x: int,
        y: int,
        scale: float = 0.125,
        cadence: int = 45,
        raw_data_dir: str | Path = RAW_DATA_DIR,
        timeseries_data_dir: str | Path = TIMESERIES_DATA_DIR,
        email: str | None = None,
        force_recompute: bool = False,
) -> tuple[Time, np.ndarray]:
    """
    Returns a Doppler velocity time series for a pixel coordinate and time
    range, downloading and/or extracting as needed, and caching the result.

    Convenience orchestrator over download_dopplergram_fits,
    extract_velocity_timeseries, and storage.*_npz; call those directly
    for finer control.

    :param start_time: Start of the time range (TAI scale).
    :param end_time: End of the time range (TAI scale).
    :param x: Horizontal pixel coordinate.
    :param y: Vertical pixel coordinate.
    :param scale: Spatial downsampling factor for raw FITS downloads.
    :param cadence: Temporal cadence, in seconds.
    :param raw_data_dir: Root directory for raw FITS downloads/cache.
    :param timeseries_data_dir: Root directory for cached extracted timeseries.
    :param email: Registered JSOC email; falls back to JSOC_EMAIL in .env.
    :param force_recompute: If True, re-extract from raw FITS even if a
        matching processed cache exists (existing raw files are still reused).
    :return: (times, velocities). Velocity unit: m/s.
    """
    require_tai(start_time)
    require_tai(end_time)

    raw_cache_dir = _ensure_raw_data(start_time, end_time, scale, cadence,
                                     Path(raw_data_dir), email)

    dataset_name = get_dataset_dir_name(scale, cadence)
    range_tag = f"{time_to_fits_str(start_time)}_{time_to_fits_str(end_time)}"
    processed_cache_file = Path(timeseries_data_dir) / f"{dataset_name}_x{x}_y{y}_{range_tag}.npz"

    if processed_cache_file.exists() and not force_recompute:
        logger.info("Loading cached velocity timeseries from %s", processed_cache_file)
        return load_timeseries_npz(processed_cache_file)

    logger.info("Extracting velocity timeseries for (x=%d, y=%d)...", x, y)
    times, velocities = extract_velocity_timeseries(raw_cache_dir, x, y, cadence)

    Path(timeseries_data_dir).mkdir(parents=True, exist_ok=True)
    save_timeseries_npz(processed_cache_file, times, velocities, cadence)
    return times, velocities