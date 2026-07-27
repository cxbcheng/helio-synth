import os
from pathlib import Path
import drms
from dotenv import load_dotenv

def get_drms_client(email: str | None = None) -> drms.Client:
    """
    Factory function to initialize and validate a DRMS Client.
    Email is taken from the JSOC_EMAIL environment variable if not provided.
    """
    if not email:
        load_dotenv()
        email = os.environ.get('JSOC_EMAIL')
    if not email:
        raise ValueError("JSOC Email is required. Provide it as an argument or set JSOC_EMAIL in .env.")

    client = drms.Client(email=email)
    return client

def download_dopplergram_fits(
    start_time: str,
    end_time: str,
    download_dir: str | Path,
    email: str | None = None,
    scale: float = 1.0,
    cadence: int = 45,
) -> Path:
    """
    Submits an export request to JSOC and downloads FITS files to download_dir based on
    the given parameters.
    The FITS files are selected from a time interval and processed by JSOC to compress
    the files by a scale factor 'scale' through boxcar averaging.
    Notice how long these export requests take!
    Each 4096x4096 FITS image is ~18 MB, thus 1 day worth of data at full resolution is ~68 GB.
    For this reason, it is recommended to scale down (scale < 1.0) to reduce the file size
    by scale^2.
    Relevant processing documentation: http://jsoc.stanford.edu/doxygen_html/group__jsoc__rebin.html
    :param start_time: Open lower bound for time interval.
    :param end_time: Open upper bound for time interval.
    :param download_dir: Directory to download FITS files. If it does not exist, one will be made.
    :param email: JSOC email.
    :param scale: The number by which to scale the FITS solar disk image.
        This effectively scales the file size of the image by a factor of scale^2.
        If scale=1.0, we have the original image size 4096x4096.
    :param cadence: Time step, i.e. the time interval between two datapoints.
        Default is 45 (seconds). Since hmi data is taken in 45 second intervals,
        cadence should be a positive integer multiple of 45.
    :return: The downloaded directory as a pathlib.Path.
    """
    client = get_drms_client(email)
    target_dir = Path(download_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create the query
    qstr = f"hmi.v_45s[{start_time}-{end_time}@{cadence}s]{{image}}"

    # Processing parameters to reduce the image size
    # http://jsoc.stanford.edu/doxygen_html/group__jsoc__rebin.html
    process = {
        'rebin': {
            'scale': scale,
            'method': 'boxcar',
        }
    }

    print(f"Submitting export request for '{qstr}'...")
    result = client.export(qstr, protocol='fits', process=process)
    result.wait(sleep=10) # Ping every 10 seconds

    print(f"\nRequest URL: {result.request_url}")
    print(f"Downloading {len(result.urls)} file(s) to '{target_dir}'...\n")
    result.download(str(target_dir))
    print("Download completed.")
    return target_dir