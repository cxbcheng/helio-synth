import re

from astropy.time import Time


@DeprecationWarning
def clean_drms_timestamp(timestamp: str) -> str:
    """
    Sanitizes DRMS timestamp strings for use in file and folder path names.
    Examples:
        "2020.01.01_00:00:00_TAI" -> "20200101_000000"
        "2020-01-01T00:00:00"     -> "20200101_000000"
    """
    ts_clean = timestamp.replace('_TAI', '')
    ts_clean = re.sub(r'[-:.]', '', ts_clean)
    ts_clean = ts_clean.replace('T', '_')
    return ts_clean


@DeprecationWarning
def make_query_name(start_time: str, end_time: str, scale: float, cadence: int) -> str:
    """
    Helper to standardize raw data folder lookups for hmi_v.45s data products.
    """
    return clean_drms_timestamp(f"hmi_v_45s_{start_time}_{end_time}_{scale}_{cadence}")


def time_to_jsoc_str(t: Time) -> str:
    """
    Formats a Time as a JSOC-style timestamp string, e.g. "2020.01.01_00:00:00_TAI".
    Used to build DRMS query strings such as "hmi.v_45s[{start}-{end}@{cadence}s]".
    :raises ValueError: if t.scale is not 'tai'.
    """
    if t.scale != 'tai':
        raise ValueError('Time scale must be "tai"')
    return f"{t.strftime('%Y.%m.%d_%H:%M:%S')}_TAI"


def get_data_dir_name(scale: float, cadence: int, product: str = 'hmi.v_45s') -> str:
    """
    Directory name for a raw dataset variant, identified by product,
    download scale, and cadence.

    :param scale: Download rebin scale (see `download_dopplergram_fits`).
    :param cadence: Sampling cadence in seconds.
    :param product: JSOC dataset series name.

    Examples:

    >>> get_data_dir_name(0.125, 45)
    'hmi.v_45s.scale0.125_cadence45s'
    >>> get_data_dir_name(1.0, 90)
    'hmi.v_45s.scale1_cadence90s'
    """
    return f"{product}.scale{scale:g}_cadence{cadence}s"


def time_to_fits_str(time: Time) -> str:
    """
    Formats a Time as a JSOC filename-style timestamp string,
    e.g. "20200101_000000_TAI" -- used in exported FITS filenames
    (contrast with time_to_jsoc_str, which uses the query-string format
    with "." and ":" separators, e.g. "2020.01.01_00:00:00_TAI").

    :raises ValueError: if time.scale is not 'tai'.

    Examples:

    >>> time_to_fits_str(Time('2020-01-01T00:00:00', scale='tai'))
    '20200101_000000_TAI'
    """
    if time.scale != 'tai':
        raise ValueError('Time scale must be "tai"')
    return f"{time.strftime('%Y%m%d_%H%M%S')}_TAI"


def get_fits_name(time: Time, prefix: str = 'hmi.v_45s', suffix: str = '2.Dopplergram.fits') -> str:
    """
    Constructs a FITS filename following the standard JSOC dataset naming convention.

    Examples:

    >>> t = Time('2020-01-01T00:00:00', scale='tai')
    >>> get_fits_name(t)
    'hmi.v_45s.20200101_000000_TAI.2.Dopplergram.fits'
    """
    timestamp = time_to_jsoc_str(time)
    clean_prefix = prefix.rstrip('.')
    clean_suffix = suffix.lstrip('.')
    return f"{clean_prefix}.{timestamp}.{clean_suffix}"