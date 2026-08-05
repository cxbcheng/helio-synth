import re
from pathlib import Path
from typing import Literal

from astropy.time import Time

from heliosynth.time_utils import require_tai


def time_to_jsoc_str(time: Time) -> str:
    """
    Formats a Time as a JSOC-style timestamp string, e.g. "2020.01.01_00:00:00_TAI".
    Used to build DRMS query strings such as "hmi.v_45s[{start}-{end}@{cadence}s]".
    """
    require_tai(time)
    return f"{time.strftime('%Y.%m.%d_%H:%M:%S')}_TAI"


def get_dataset_dir_name(scale: float, cadence: int, product: str = 'hmi.v_45s') -> str:
    """
    Gets the path to the directory containing FITS files unique to the
    given parameters.

    :param scale: Download rebin scale (see `download_dopplergram_fits`).
    :param cadence: Sampling cadence in seconds.
    :param product: JSOC dataset series name.

    Examples:

    >>> get_dataset_dir_name(0.125, 45)
    'hmi.v_45s.scale0.125_cadence45s'
    >>> get_dataset_dir_name(1.0, 90)
    'hmi.v_45s.scale1_cadence90s'
    """
    return f"{product}.scale{scale:g}_cadence{cadence}s"


def time_to_fits_str(time: Time) -> str:
    """
    Formats a Time as a JSOC filename-style timestamp string,
    e.g. "20200101_000000_TAI" -- used in exported FITS filenames
    (contrast with time_to_jsoc_str, which uses the query-string format
    with "." and ":" separators, e.g. "2020.01.01_00:00:00_TAI").

    Examples:

    >>> time_to_fits_str(Time('2020-01-01T00:00:00', scale='tai'))
    '20200101_000000_TAI'
    """
    require_tai(time)
    return f"{time.strftime('%Y%m%d_%H%M%S')}_TAI"


def get_fits_name(time: Time, prefix: str = 'hmi.v_45s', suffix: str = '2.Dopplergram.fits') -> str:
    """
    Constructs a FITS filename following the standard JSOC dataset naming convention.
    This function is effectively a helper for formatting astropy.time.Time as
    strings matching the JSOC filename format.

    Examples:

    >>> t = Time('2020-01-01T00:00:00', scale='tai')
    >>> get_fits_name(t)
    'hmi.v_45s.20200101_000000_TAI.2.Dopplergram.fits'
    """
    require_tai(time)
    timestamp = time_to_fits_str(time)
    clean_prefix = prefix.rstrip('.')
    clean_suffix = suffix.lstrip('.')
    return f"{clean_prefix}.{timestamp}.{clean_suffix}"


def get_dataset_dir(raw_data_dir: str | Path, scale: float, cadence: int, product: str = 'hmi.v_45s') -> Path:
    """
    Gets the path to the directory containing FITS files unique to the
    given parameters. `raw_data_dir` is the general directory containing
    this directory.
    """
    return Path(raw_data_dir) / get_dataset_dir_name(scale=scale, cadence=cadence, product=product)


def get_fits_path(dataset_dir: str | Path, time: Time, prefix: str = 'hmi.v_45s', suffix: str = '2.Dopplergram.fits') -> Path:
    """
    Gets the path to a FITS file from a dataset directory (i.e. one containing
    FITS files) by getting the standard name for a FITS file unique to a time.
    This function is effectively a helper for formatting astropy.time.Time as
    strings matching the JSOC filename format.
    """
    return Path(dataset_dir) / get_fits_name(time=time, prefix=prefix, suffix=suffix)


def fits_name_to_time(filename: str) -> Time:
    """
    Inverse of get_fits_name/time_to_fits_str: parses the observation time
    out of a standard JSOC-convention FITS filename, e.g.
    "hmi.v_45s.20200101_000000_TAI.2.Dopplergram.fits" -> Time for
    2020-01-01T00:00:00 TAI.
    :raises ValueError: if no "<YYYYMMDD>_<HHMMSS>_TAI" segment is found.
    """
    match = re.search(r'(\d{8})_(\d{6})_TAI', filename)
    if not match:
        raise ValueError(f"Could not find a JSOC-style timestamp in '{filename}'")
    date_part, time_part = match.groups()
    iso = (f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}T"
           f"{time_part[0:2]}:{time_part[2:4]}:{time_part[4:6]}")
    return Time(iso, format='isot', scale='tai')


def disk_velocity_zarr_name(n_points: int, scale: float, cadence: int, product: str = 'hmi.v_45s'):
    """
    Constructs a disk velocity zarr filename unique to the given parameters.
    n_points indicates the number of sample points from the disk.
    """
    return f"{get_dataset_dir_name(scale, cadence, product)}_n{n_points}.zarr"


def disk_velocity_zarr_path(
    data_dir: Path,
    n_points: int,
    scale: float,
    cadence: int,
    product: str = 'hmi.v_45s'
) -> Path:
    """
    Constructs a disk velocity zarr path unique to the given parameters.
    n_points indicates the number of sample points from the disk.
    """
    return data_dir / disk_velocity_zarr_name(n_points, scale, cadence, product)


def doppler_image_path(dataset_dir: Path, time: Time, format: Literal['webp', 'png', 'jpg'] = 'webp'):
    """
    Constructs a Doppler image path unique to the timestamp and file format.
    """
    require_tai(time)
    return dataset_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.{format}"


def covered_intervals(
    dataset_dir: Path,
    cadence: float,
    gap_tolerance: float = 2.5
) -> list[tuple[Time, Time]]:
    """
    Contiguous time intervals covered by existing FITS files inside `dataset_dir`,
    determined from filenames alone (fits_name_to_time). Files separated
    by <= gap_tolerance * cadence are merged into one interval (tolerating
    the normal handful of dropped frames); larger gaps start a new,
    disjoint interval. Returns [] if raw_dir has no FITS files.
    """
    fits_files = list(dataset_dir.glob('*.fits')) if dataset_dir.exists() else []
    if not fits_files:
        return []

    times = sorted(fits_name_to_time(f.name) for f in fits_files)
    intervals = []
    seg_start = seg_end = times[0]
    for t in times[1:]:
        if (t - seg_end).sec <= gap_tolerance * cadence:
            seg_end = t
        else:
            intervals.append((seg_start, seg_end))
            seg_start = seg_end = t
    intervals.append((seg_start, seg_end))
    return intervals


def missing_subranges(
    dataset_dir: Path,
    requested_start: Time,
    requested_end: Time,
    cadence: float,
    gap_tolerance: float = 2.5,
) -> list[tuple[Time, Time]]:
    """
    Sub-ranges of [requested_start, requested_end] not already covered by
    existing FITS files from `dataset_dir` -- i.e. the time ranges that
    need to be downloaded.
    """
    covered = covered_intervals(dataset_dir, cadence, gap_tolerance)

    # Restrict coverage to the requested interval
    clipped = sorted(
        (max(s, requested_start), min(e, requested_end))
        for s, e in covered
        if s < requested_end and e > requested_start
    )

    missing = []

    cursor = requested_start
    for s, e in clipped:
        if s > cursor:
            missing.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < requested_end:
        missing.append((cursor, requested_end))
    return missing