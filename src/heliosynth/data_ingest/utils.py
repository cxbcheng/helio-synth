from heliosynth.path_utils import fits_name_to_time

from pathlib import Path

from astropy.time import Time


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