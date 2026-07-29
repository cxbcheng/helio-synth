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

def make_query_name(start_time: str, end_time: str, scale: float, cadence: int) -> str:
    """
    Helper to standardize raw data folder lookups for hmi_v.45s data products.
    """
    return clean_drms_timestamp(f"hmi_v_45s_{start_time}_{end_time}_{scale}_{cadence}")