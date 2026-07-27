import os
import tarfile
from pathlib import Path
import drms
import pandas as pd
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
    poll_interval: int | None = 10,
) -> Path:
    """
    Submits an export request to JSOC and downloads FITS files to download_dir based on
    the given parameters.
    The FITS files are selected from a time interval and processed by JSOC to compress
    the files by a scale factor 'scale' through boxcar averaging.
    The request is first downloaded as a tar file then extracted into FITS afterward.
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
    :param poll_interval: Time in seconds between export status updates.
        If None, the JSOC server supplied value is used.
    :return: The downloaded directory as a pathlib.Path.
    """
    # Input validation
    if scale <= 0:
        raise ValueError("scale must be positive")

    if cadence <= 0 or cadence % 45 != 0:
        raise ValueError("cadence must be a positive multiple of 45")

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
    result = client.export(qstr, method='url-tar', protocol='fits', process=process)
    result.wait(sleep=poll_interval)

    print(f"Request URL: {result.request_url}")
    print(f"Downloading {len(result.urls)} file(s) to '{target_dir}'...")
    tars = result.download(str(target_dir))

    failed = _extract_tars(tars, target_dir)
    if failed:
        print(f"Warning: {len(failed)} archive(s) failed to extract: {', '.join(failed)}")

    print(f"Download completed. {len(list(target_dir.glob('*.fits')))} file(s) ready.")
    return target_dir


def _extract_tars(tars: pd.DataFrame, target_dir: Path) -> list[str]:
    """
    Extract all downloaded TAR archives into target_dir and remove the
    archives on success.

    :param tars: The DataFrame returned by drms.ExportRequest.download(),
        expected to have a 'download' column of local file paths (NaN for
        entries that failed to download).
    :param target_dir: Directory containing the downloaded tar files and
        where their contents will be extracted.
    :return: List of tar filenames that failed to extract (empty if all
        succeeded).
    """
    failed: list[str] = []

    missing = tars['download'].isna().sum()
    if missing:
        print(f"Warning: {missing} file(s) failed to download and will be skipped.")

    for tar_path_str in tars['download'].dropna():
        tar_path = Path(tar_path_str)

        if not tar_path.is_file():
            print(f"Skipping {tar_path.name}: file not found.")
            failed.append(tar_path.name)
            continue

        if not tarfile.is_tarfile(tar_path):
            print(f"Skipping {tar_path.name}: not a valid tar file.")
            failed.append(tar_path.name)
            continue

        try:
            print(f"Extracting {tar_path.name}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=target_dir, filter='data')
        except (tarfile.TarError, OSError) as e:
            print(f"Error extracting {tar_path.name}: {e}")
            failed.append(tar_path.name)
            continue

        try:
            tar_path.unlink()
        except OSError:
            print(f"Could not remove {tar_path.name}")

    return failed