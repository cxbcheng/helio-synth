import json
import logging
import os
import tarfile
from pathlib import Path
import drms
import pandas as pd
from astropy.time import Time
from dotenv import load_dotenv
from tqdm import tqdm

from heliosynth.path_utils import time_to_jsoc_str, get_dataset_dir
from heliosynth.time_utils import require_tai

logger = logging.getLogger(__name__)

STATUS_FILENAME = 'download_status.json'
_STATUS_SAVE_INTERVAL = 50  # persist extraction progress every N members


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


def _status_path(dataset_dir: Path) -> Path:
    return dataset_dir / STATUS_FILENAME


def _load_status(dataset_dir: Path) -> dict | None:
    """
    Loads the download-status JSON for dataset_dir, if present. Returns
    None if no status file exists, or if one exists but can't be parsed
    (treated the same as "no status" rather than raising, so a corrupt
    status file doesn't block a retry).
    """
    path = _status_path(dataset_dir)
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read status file %s: %s -- treating as absent.", path, e)
        return None


def _save_status(dataset_dir: Path, status: dict) -> None:
    """
    Writes the download-status JSON atomically (temp file + rename), so
    an interrupted write can never leave a corrupt/partial status file.
    """
    path = _status_path(dataset_dir)
    tmp_path = path.with_suffix('.tmp.json')
    with open(tmp_path, 'w') as f:
        json.dump(status, f, indent=2)
    os.replace(tmp_path, path)


def download_dopplergram_fits(
        start_time: Time,
        end_time: Time,
        raw_data_dir: str | Path | None = None,
        email: str | None = None,
        scale: float | None = None,
        cadence: int = 45,
        poll_interval: int | None = 10,
        request_id: str | None = None,
        dataset_dir: Path | None = None,
        resolution: int | None = None,
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

    Progress is recorded to a "download_status.json" file in dataset_dir
    after the JSOC request succeeds, after tar files are downloaded, and
    once extraction completes. Extracted tar files are kept on disk
    (not deleted) so a re-run can detect and resume partial extraction
    by comparing each tar member's size against any file already present.
    Re-running this function on an already-complete or in-progress
    dataset_dir will skip whatever steps are already done rather than
    restarting from scratch.

    :param start_time: Open lower bound for time interval.
    :param end_time: Open upper bound for time interval.
    :param raw_data_dir: Raw data directory. Another directory will be created
        underneath unique to the resolution and cadence. Specify `dataset_dir`
        to download directly to a specified directory.
    :param email: JSOC email.
    :param scale: The number by which to scale the FITS solar disk image.
        This effectively scales the file size of the image by a factor of scale^2.
        If scale=1.0, we have the original image size 4096x4096. If neither
        scale nor resolution are specified, then the full resolution is used.
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
        If not given, a request ID saved in dataset_dir's status file
        (from a previous interrupted run) is used automatically if present.
    :param dataset_dir: Download directory. Should not be specified if
        raw_data_dir is already specified.
    :param resolution: Resolution/width of the FITS solar disk image.
        The full resolution is 4096 x 4096. Should not be specified if
        scale is already specified, as one is calculated from the other.
    :return: The directory containing downloaded FITS files, as a Path.
    """
    require_tai(start_time)
    require_tai(end_time)
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time.isot}) must be strictly before end_time ({end_time.isot})")
    if cadence <= 0 or cadence % 45 != 0:
        raise ValueError("cadence must be a positive multiple of 45")
    if dataset_dir is not None and raw_data_dir is not None:
        raise ValueError("either dataset_dir or raw_data_dir should be specified, not both")
    if dataset_dir is None and raw_data_dir is None:
        raise ValueError("either dataset_dir or raw_data_dir should be specified")
    if scale is not None and resolution is not None:
        raise ValueError("either scale or resolution should be specified, not both")
    if scale is None and resolution is None:
        scale = 1.0

    if scale is not None:
        resolution = round(4096 * scale)
    else:
        scale = resolution / 4096

    if scale <= 0:
        raise ValueError(f"scale/resolution must be positive")
    if resolution > 4096:
        raise ValueError(f"resolution is larger than full resolution HMI image: {resolution}")

    if dataset_dir is None:
        dataset_dir = get_dataset_dir(raw_data_dir, resolution=resolution, cadence=cadence)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    status = _load_status(dataset_dir)

    if status is None:
        # Backward compatibility: a dataset_dir from before status tracking
        # existed may already have tar files with no status.json. At
        # least avoid re-submitting/re-downloading from JSOC in that case.
        existing_tars = sorted(dataset_dir.glob('*.tar'))
        if existing_tars:
            logger.info(
                "Found %d existing .tar file(s) with no status record; "
                "skipping JSOC request/download and resuming extraction.",
                len(existing_tars),
            )
            status = {'status': 'downloaded', 'request_id': None, 'request_url': None, 'tars': {}}

    if status is not None and status.get('status') == 'complete':
        n_fits = len(list(dataset_dir.glob('*.fits')))
        if n_fits > 0:
            logger.info("Dataset already complete (%d FITS file(s)) at %s; nothing to do.", n_fits, dataset_dir)
            return dataset_dir
        logger.warning(
            "Status file marks this dataset complete, but no FITS files "
            "were found in %s -- proceeding as if incomplete.", dataset_dir,
        )

    if status is None or status.get('status') not in ('downloaded', 'extracting'):
        client = get_drms_client(email)

        effective_request_id = request_id
        if effective_request_id is None and status is not None:
            effective_request_id = status.get('request_id')

        if effective_request_id is not None:
            logger.info("Resuming export request '%s'...", effective_request_id)
            result = client.export_from_id(effective_request_id)
        else:
            start_time_str = time_to_jsoc_str(start_time)
            end_time_str = time_to_jsoc_str(end_time)
            qstr = f"hmi.v_45s[{start_time_str}-{end_time_str}@{cadence}s]{{image}}"
            process = {'rebin': {'scale': scale, 'method': 'boxcar'}}
            logger.info("Submitting export request for '%s'...", qstr)
            result = client.export(qstr, method='url-tar', protocol='fits', process=process)

        logger.info("Resolution size: %d x %d", resolution, resolution)
        logger.info(
            "Request ID: %s (pass as request_id=%r to resume if interrupted)",
            result.id, result.id,
        )

        # Checkpoint 1: request has succeeded and an id is known, recorded
        # before wait()/download() -- so a crash during the (often long)
        # JSOC processing wait still leaves enough to resume from.
        status = {
            'status': 'requested',
            'request_id': result.id,
            'request_url': result.request_url,
            'tars': {},
        }
        _save_status(dataset_dir, status)

        result.wait(sleep=poll_interval)

        logger.info("Request URL: %s", result.request_url)
        logger.info("Downloading %d file(s) to '%s'...", len(result.urls), dataset_dir)
        tars_df = result.download(str(dataset_dir))

        missing = tars_df['download'].isna().sum()
        if missing:
            logger.warning("%d file(s) failed to download and will be skipped.", missing)

        # Checkpoint 2: tar files downloaded, recorded before extraction.
        # NOTE: relies on drms's own download() skipping already-present
        # local files when resuming a request whose tars partially
        # downloaded before -- worth confirming against your installed
        # drms version if you hit unexpected re-downloads here.
        status['status'] = 'downloaded'
        for downloaded_path in tars_df['download'].dropna():
            status['tars'][Path(downloaded_path).name] = {
                'extracted': False, 'n_members': None, 'n_extracted': 0,
            }
        _save_status(dataset_dir, status)
    else:
        logger.info("Tar file(s) already downloaded for %s; skipping JSOC request.", dataset_dir)

    status['status'] = 'extracting'
    _save_status(dataset_dir, status)

    tar_paths = sorted(dataset_dir.glob('*.tar'))
    failed = _extract_tars(tar_paths, dataset_dir, status)

    if not failed:
        status['status'] = 'complete'
    else:
        # status stays 'extracting' -- a re-run will retry only the
        # failed/incomplete tars, not re-request or re-download anything.
        logger.warning(
            "%d archive(s) failed to extract: %s. Re-run to retry "
            "extraction -- already-downloaded tars will not be re-requested.",
            len(failed), ', '.join(failed),
        )
    _save_status(dataset_dir, status)

    logger.info("Download completed. %d file(s) ready.", len(list(dataset_dir.glob('*.fits'))))
    return dataset_dir


def _extract_tars(tar_paths: list[Path], target_dir: Path, status: dict) -> list[str]:
    """
    Extract all downloaded TAR archives into target_dir.

    Also detects and removes stray ``*.tar.part`` files -- leftover
    partial downloads from a previous interrupted run.

    Tar files are intentionally NOT deleted after extraction. They stay
    on disk as the reference used to detect and resume partially-extracted
    archives: each member's expected size (from the tar) is compared
    against any file already on disk at that path -- a match is skipped,
    anything missing or size-mismatched is (re-)extracted. This check is
    authoritative on its own, so resuming is correct even if `status`
    wasn't fully up to date when a previous run was interrupted; the
    periodic `status` saves below are for progress visibility, not
    a requirement for correct resumption.

    :param tar_paths: Tar files to extract (existing files in target_dir).
    :param target_dir: Directory containing the downloaded tar files and
        where their contents will be extracted.
    :param status: Mutable download-status dict; its 'tars' entry is
        updated in place with per-tar extraction progress and persisted
        periodically via _save_status.
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

    tars = status.setdefault('tars', {})

    for tar_path in tar_paths:
        tar_name = tar_path.name
        tar_status = tars.setdefault(tar_name, {'extracted': False, 'n_members': None, 'n_extracted': 0})

        if tar_status.get('extracted'):
            continue  # already fully extracted on a previous run

        if not tar_path.is_file():
            logger.warning("Skipping %s: file not found.", tar_name)
            failed.append(tar_name)
            continue

        if not tarfile.is_tarfile(tar_path):
            logger.warning("Skipping %s: not a valid tar file.", tar_name)
            failed.append(tar_name)
            continue

        n_extracted = 0
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                tar_status['n_members'] = len(members)
                for member in tqdm(members, desc=f'Extracting {tar_name}', unit='file'):
                    target_path = target_dir / member.name
                    already_done = target_path.is_file() and target_path.stat().st_size == member.size
                    if not already_done:
                        tar.extract(member, path=target_dir, filter='data')
                    n_extracted += 1
                    if n_extracted % _STATUS_SAVE_INTERVAL == 0:
                        tar_status['n_extracted'] = n_extracted
                        _save_status(target_dir, status)
        except (tarfile.TarError, OSError) as e:
            tar_status['n_extracted'] = n_extracted
            _save_status(target_dir, status)
            logger.warning("Error extracting %s: %s", tar_name, e)
            failed.append(tar_name)
            continue

        tar_status['n_extracted'] = n_extracted
        tar_status['extracted'] = True
        _save_status(target_dir, status)

    return failed
