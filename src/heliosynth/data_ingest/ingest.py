import logging
from pathlib import Path

import numpy as np
from astropy.time import Time

from heliosynth.data_ingest.client import download_dopplergram_fits
from heliosynth.data_ingest.extraction import extract_velocity_timeseries
from heliosynth.data_ingest.storage import load_timeseries_npz, save_timeseries_npz
from heliosynth.data_ingest.utils import get_dataset_dir_name, get_dataset_dir, missing_subranges, time_to_fits_str
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
        gap_tolerance: int = 2.5,
        force_recompute: bool = False,
        download_missing: bool = False,
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
    :param raw_data_dir: Root directory for raw FITS downloads (e.g. 'data/raw').
    :param timeseries_data_dir: Directory for cached extracted timeseries.
    :param email: Registered JSOC email; falls back to JSOC_EMAIL in .env.
    :param gap_tolerance: Tolerance for missing data in the requested time range.
        Does not do anything if `download_missing` is set to `False`.
        See `download_missing_fits` for details.
    :param force_recompute: If True, re-extract from raw FITS even if a
        matching processed cache exists (existing raw files are still reused).
    :param download_missing: If set to `True`, looks for missing data in
        the raw data directory and downloads it (via `download_missing_fits`).
    :return: (times, velocities). Velocity unit: m/s.
    """
    require_tai(start_time)
    require_tai(end_time)

    if download_missing:
        download_missing_fits(
            start_time=start_time,
            end_time=end_time,
            scale=scale,
            cadence=cadence,
            gap_tolerance=gap_tolerance,
            raw_data_dir=raw_data_dir,
            email=email)

    dataset_dir = get_dataset_dir(raw_data_dir, scale, cadence)
    dataset_name = get_dataset_dir_name(scale, cadence)  # For cache file name
    range_tag = f"from{time_to_fits_str(start_time)}_to{time_to_fits_str(end_time)}"
    timeseries_cache_file = Path(timeseries_data_dir) / f"{dataset_name}_x{x}_y{y}_{range_tag}.npz"
    logger.info('%s', timeseries_cache_file)

    if timeseries_cache_file.exists() and not force_recompute:
        logger.info("Loading cached velocity timeseries from %s", timeseries_cache_file)
        return load_timeseries_npz(timeseries_cache_file)

    logger.info("Extracting velocity timeseries for (x=%d, y=%d)...", x, y)
    times, velocities = extract_velocity_timeseries(
        data_dir=dataset_dir,
        x=x,
        y=y,
        cadence=cadence,
        start_time=start_time,
        end_time=end_time)

    Path(timeseries_data_dir).mkdir(parents=True, exist_ok=True)
    save_timeseries_npz(timeseries_cache_file, times, velocities, cadence)
    return times, velocities


def download_missing_fits(
    start_time: Time,
    end_time: Time,
    scale: float,
    cadence: int,
    gap_tolerance: int = 2.5,
    raw_data_dir: Path = RAW_DATA_DIR,
    email: str | None = None,
) -> None:
    """
    Ensures FITS covering [start_time, end_time] exist locally,
    downloading each missing sub-range as its own request. Handles
    disjoint existing coverage correctly -- e.g. requesting a range far
    from previously-downloaded data downloads only that range, never the
    span between them.
    """
    raw_data_dir = get_dataset_dir(raw_data_dir, scale, cadence)
    gaps = missing_subranges(
        raw_dir=raw_data_dir,
        requested_start=start_time,
        requested_end=end_time,
        cadence=cadence,
        gap_tolerance=gap_tolerance)

    if len(gaps) == 0:
        logger.info("No missing sub-ranges found in %s", raw_data_dir)

    for gap_start, gap_end in gaps:
        logger.info("Downloading missing range %s to %s", gap_start.isot, gap_end.isot)
        download_dopplergram_fits(
            start_time=gap_start,
            end_time=gap_end,
            raw_data_dir=raw_data_dir,
            email=email,
            scale=scale,
            cadence=cadence)