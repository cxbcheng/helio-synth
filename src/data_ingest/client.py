import os
from pathlib import Path

import drms
import numpy as np
from dotenv import load_dotenv

def get_drms_client(email: str | None = None) -> drms.Client:
    """
    Factory function to initialize and validate a DRMS Client.
    """
    if not email:
        load_dotenv()
        email = os.environ.get('JSOC_EMAIL')
    if not email:
        raise ValueError("JSOC Email is required. Provide it as an argument or set JSOC_EMAIL in .env.")

    client = drms.Client(email=email)
    return client

def extract_velocity_timeseries(
    start_time: str,
    end_time: str,
    x: float,
    y: float,
    out_dir: str | Path,
    email: str | None = None,
    tracking: bool = True,
    loc_units: str = 'arcsec',
    box_units: str = 'pixels',
    cutout_length: float = 1.0,
) -> np.ndarray:
    """
    TODO: documentation
    :param start_time:
    :param end_time:
    :param x:
    :param y:
    :param out_dir:
    :param email:
    :param tracking: Set True to track solar rotation, False otherwise.
    :param loc_units: Units for the coordinates (x, y) as defined in the helioprojective frame
        of SDO. Default is 'arcsec'.
        Documentation: https://jsoc.stanford.edu/doxygen_html/group__im__patch.html
    :param box_units: Units for the cutout patch. Default is 'pixels'.
        Documentation: https://jsoc.stanford.edu/doxygen_html/group__im__patch.html
    :param cutout_length: The length and width of the square cutout for the FITS solar disk image.
        This cuts down the size of the image by a factor of cutout_length^2. If cutout_length=1.0,
        we have the original image size 4096x4096.
    :return:
    """
    client = get_drms_client(email)

    # Set up download directory
    # Clean parameter string representation for directory naming
    start_clean = start_time.replace('.', '').replace(':', '').replace('_TAI', '')
    end_clean = end_time.replace('.', '').replace(':', '').replace('_TAI', '')

    # Cache directory path based on query parameters
    query_folder_name = f"hmi_v_{start_clean}_{end_clean}_x{round(x)}_y{round(y)}_{round(cutout_length)}"
    cache_dir = Path(out_dir) / query_folder_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # TODO: check if data is already stored locally

    # Create the query
    qstr = f"hmi.v_45s[{start_time}-{end_time}]{{image}}"

    # Processing parameters to create a rectangular cutout
    # Documentation: https://jsoc.stanford.edu/doxygen_html/group__im__patch.html
    process = {
        'im_patch': {
            't_ref': start_time,        # time when (x, y) is centered
            't': 0 if tracking else 1,  # "tracking disabled" flag => 0 is enabled
            'r': 0,
            'c': 0,
            'x': x,
            'y': y,
            'locunits': loc_units,
            'boxunits': box_units,
            'width': cutout_length,
            'height': cutout_length,
        }
    }

    print(f"Submitting export request for '{qstr}' at ({x}, {y})...")
    result = client.export(qstr, protocol='fits', process=process)
    result.wait(sleep=10) # Ping every 10 seconds

    print(f"\nRequest URL: {result.request_url}")
    print(f"{len(result.urls)} file(s) available for download.\n")
    result.download(str(cache_dir))
    print(f"Data downloaded in {cache_dir}.\n")

    # TODO: return velocity timeseries instead
    return result.data

# FIXME: remove tests
# Note how long these export requests take! Make sure to implement cutouts so that we don't try to
# extract the entire 4096x4096 image, even over the course of a day (which is about 3840 images).
# Each 4096x4096 FITS image is ~18 MB => 1 day worth of data at full resolution is ~68 GB.
# For this reason, we will grab 64x64 cutouts to cut the file size down by a factor of 64^2,
# thereby cutting down 72 days worth of data to ~1-2 GB.
extract_velocity_timeseries(
    start_time='2020-01-01T00:00:00Z',
    end_time='2020-01-01T00:10:00Z',
    x=0,
    y=0,
    cutout_length=64.0,
    out_dir=os.environ.get("DATA_OUT_DIR"),
)