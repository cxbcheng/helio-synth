import logging
import os
import tarfile
from pathlib import Path
import drms
import pandas as pd
from astropy.time import Time
from dotenv import load_dotenv

from heliosynth.path_utils import time_to_jsoc_str, get_dataset_dir
from heliosynth.time_utils import require_tai

logger = logging.getLogger(__name__)


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
    start_time: Time,
    end_time: Time,
    raw_data_dir: str | Path,
    email: str | None = None,
    scale: float = 1.0,
    cadence: int = 45,
    poll_interval: int | None = 10,
    request_id: str | None = None,
) -> Path:
    """
    Submits an export request to JSOC and downloads FITS files under
    the directory {raw_data_dir}/hmi.45s.scale{scale}_cadence{cadence}s
    based on the given parameters.

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
    :param raw_data_dir: Raw data directory.
    :param email: JSOC email.
    :param scale: The number by which to scale the FITS solar disk image.
        This effectively scales the file size of the image by a factor of scale^2.
        If scale=1.0, we have the original image size 4096x4096.
    :param cadence: Time step, i.e. the time interval between two datapoints.
        Default is 45 (seconds). Since hmi data is taken in 45 second intervals,
        cadence should be a positive integer multiple of 45.
    :param poll_interval: Time in seconds between export status updates.
        If None, the JSOC server supplied value is used.
    :param request_id: An existing JSOC export request ID to resume from,
        instead of submitting a new export request. Use this to recover
        from an interrupted download without resubmitting an identical
        request and re-queueing at JSOC -- the request ID is always
        logged (see below) specifically so it's available for this.
    :return: The directory containing downloaded FITS files, as a Path.
    """
    require_tai(start_time)
    require_tai(end_time)
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time.isot}) must be strictly before end_time ({end_time.isot})")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if cadence <= 0 or cadence % 45 != 0:
        raise ValueError("cadence must be a positive multiple of 45")
    resolution = round(4096 * scale)
    client = get_drms_client(email)
    target_dir = get_dataset_dir(raw_data_dir, resolution=resolution, cadence=cadence)
    target_dir.mkdir(parents=True, exist_ok=True)

    if request_id is not None:
        logger.info("Resuming export request '%s'...", request_id)
        result = client.export_from_id(request_id)
    else:
        start_time_str = time_to_jsoc_str(start_time)
        end_time_str = time_to_jsoc_str(end_time)
        qstr = f"hmi.v_45s[{start_time_str}-{end_time_str}@{cadence}s]{{image}}"
        process = {'rebin': {'scale': scale, 'method': 'boxcar'}}
        logger.info("Submitting export request for '%s'...", qstr)
        result = client.export(qstr, method='url-tar', protocol='fits', process=process)

    # Logs immediately before the request can fail
    logger.info(
        "Request ID: %s (pass as request_id=%r to resume if interrupted)",
        result.id, result.id,
    )

    result.wait(sleep=poll_interval)

    logger.info("Request URL: %s", result.request_url)
    logger.info("Downloading %d file(s) to '%s'...", len(result.urls), target_dir)
    tars = result.download(str(target_dir))

    failed = _extract_tars(tars, target_dir)
    if failed:
        logger.warning(
            "%d archive(s) failed to extract: %s. Re-run with "
            "request_id=%r to retry the download without resubmitting.",
            len(failed), ', '.join(failed), result.id,
        )

    logger.info("Download completed. %d file(s) ready.", len(list(target_dir.glob('*.fits'))))
    return target_dir


def _extract_tars(tars: pd.DataFrame, target_dir: Path) -> list[str]:
    """
    Extract all downloaded TAR archives into target_dir and remove the
    archives on success.

    Also detects and removes stray ``*.tar.part`` files -- leftover
    partial downloads from a previous interrupted run.

    :param tars: The DataFrame returned by drms.ExportRequest.download(),
        expected to have a 'download' column of local file paths (NaN for
        entries that failed to download).
    :param target_dir: Directory containing the downloaded tar files and
        where their contents will be extracted.
    :return: List of tar filenames that failed to extract or were found
        incomplete (empty if all succeeded).
    """
    failed: list[str] = []

    stray_parts = list(target_dir.glob('*.tar.part'))
    if stray_parts:
        logger.warning(
            "%d incomplete download(s) (*.tar.part) found in %s -- "
            "leftover from an interrupted download. Removing them; the "
            "corresponding file(s) will be re-downloaded on retry.",
            len(stray_parts), target_dir,
        )
        for part in stray_parts:
            try:
                part.unlink()
            except OSError as e:
                logger.warning("Could not remove stray partial file %s: %s", part.name, e)
            failed.append(part.name)

    missing = tars['download'].isna().sum()
    if missing:
        logger.warning("%d file(s) failed to download and will be skipped.", missing)

    for tar_path_str in tars['download'].dropna():
        tar_path = Path(tar_path_str)

        if not tar_path.is_file():
            logger.warning(
                "Skipping %s: file not found (may indicate an interrupted "
                "download not caught above).", tar_path.name,
            )
            failed.append(tar_path.name)
            continue

        if not tarfile.is_tarfile(tar_path):
            logger.warning("Skipping %s: not a valid tar file.", tar_path.name)
            failed.append(tar_path.name)
            continue

        try:
            logger.info("Extracting %s...", tar_path.name)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=target_dir, filter='data')
        except (tarfile.TarError, OSError) as e:
            logger.warning("Error extracting %s: %s", tar_path.name, e)
            failed.append(tar_path.name)
            continue

        try:
            tar_path.unlink()
        except OSError:
            logger.warning("Could not remove %s", tar_path.name)

    return failed