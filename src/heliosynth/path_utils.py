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


def get_dataset_dir_name(resolution: int, cadence: int, product: str = 'hmi.v_45s') -> str:
    """
    Gets the path to the directory containing FITS files unique to the
    given parameters.

    :param resolution: Image resolution of the FITS file,
        i.e. width of the (square) image.
        The full resolution of an HMI image is 4096x4096,
        thus resolution = scale * 4096.
    :param cadence: Sampling cadence in seconds.
    :param product: JSOC dataset series name.

    Examples:

    >>> get_dataset_dir_name(512, 45)
    'hmi.v_45s.res512_cadence45s'
    >>> get_dataset_dir_name(4096, 90)
    'hmi.v_45s.res4096_cadence90s'
    """
    return f"{product}.res{resolution:g}_cadence{cadence}s"


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


def get_dataset_dir(data_dir: str | Path, resolution: int, cadence: int, product: str = 'hmi.v_45s') -> Path:
    """
    Gets the path to the directory containing FITS files unique to the
    given parameters. `data_dir` is the general directory containing
    this directory, such as 'data/raw' or 'data/processed/datasets'.
    """
    return Path(data_dir) / get_dataset_dir_name(resolution=resolution, cadence=cadence, product=product)


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


def disk_velocity_zarr_name(n_points: int):
    """
    Constructs a disk velocity zarr filename.
    n_points indicates the number of sample points from the disk.
    """
    return f"n{n_points}.zarr"


def disk_velocity_zarr_path(dataset_dir: Path, n_points: int) -> Path:
    """
    Constructs a disk velocity zarr path.
    n_points indicates the number of sample points from the disk.
    """
    return dataset_dir / 'timeseries' / disk_velocity_zarr_name(n_points)


def doppler_image_path(
        dataset_dir: Path,
        time: Time,
        detrend_order: int | None = None,
        colormap: str = 'RdBu_r',
        format: Literal['webp', 'png', 'jpg'] = 'webp'
) -> Path:
    """
    Constructs a Doppler image path unique to the timestamp, colormap, and file format.
    Path is identified by dataset_dir / colormap / detrend_order / image_file
    (the detrend_order directory is named 'raw').
    """
    require_tai(time)
    order_name = f"{detrend_order}" if detrend_order is not None else 'raw'
    return dataset_dir / 'images' / colormap / order_name / f"{time.strftime('%Y%m%d_%H%M%S')}.{format}"
