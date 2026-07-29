import re


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