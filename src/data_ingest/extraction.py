from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time, TimeDelta


def extract_velocity_timeseries(
        data_dir: str | Path,
        x: int,
        y: int,
        cadence: int,
        max_files: int | None = None,
) -> tuple[Time, np.ndarray]:
    """
    Parses local FITS files in the provided directory and extracts a velocity timeseries
    for a pixel coordinate (x, y) on the solar disk image relative to the scale specified
    in the FITS file. Full resolution is 4096x4096; hence, valid positions include
    x, y \in [0, int(4096*scale)-1].
    :param data_dir: Directory containing FITS files
    :param x: x pixel position
    :param y: y pixel position
    :param cadence: The expected time step or sampling interval (seconds) between any two
        data points. The time of the image is cross-validated with the cadence to mark
        any missing data with NaNs.
    :param max_files:
    :return: (times, velocities). A uniform-cadence timeseries. Velocity unit: m/s.
        velocities[i] is NaN whenever no valid sample was found for that time slot.
    :raises ValueError: if (x, y) is out of bounds, or no valid samples are found.
    """
    target_dir = Path(data_dir)
    fits_files = sorted(list(target_dir.glob('*.fits')))

    if max_files is not None:
        fits_files = fits_files[:max_files]

    t_recs = []
    velocities = []

    for fits_file in fits_files:
        try:
            t_rec, vel = _read_velocity_sample(fits_file, x, y)
        except (OSError, KeyError, IndexError) as e:
            print(f"Skipping {fits_file.name}: {e}")
            continue
        t_recs.append(t_rec)
        velocities.append(vel)

    if not t_recs:
        raise ValueError(f"No valid samples found in {target_dir}")

    # Guard to ensure once more that the times are sorted chronologically (in case of invariance in file names)
    times_unsorted = Time(np.array(t_recs), format='isot', scale='tai')
    order = np.argsort(times_unsorted.jd)
    times = times_unsorted[order]
    velocities = np.array(velocities, dtype=np.float32)[order]

    return _align_to_cadence(times, velocities, cadence)


def _align_to_cadence(
        times: Time,
        velocities: np.ndarray,
        cadence: float,
) -> tuple[Time, np.ndarray]:
    """
    Resamples (times, velocities) onto a uniform grid spaced `cadence` seconds apart,
    spanning [times[0], times[-1]]. Assumes `times` is already sorted ascending.

    Each observed sample is assigned to a grid slot with its matching time value;
    slots with no matching sample are left as NaN.

    :param times: Sorted observation times.
    :param velocities: Velocities corresponding to `times`, same length.
    :param cadence: Grid spacing in seconds.
    :return: (grid_times, grid_velocities), uniformly spaced at `cadence`.
    """
    t0 = times[0]

    # Seconds since the first sample for each observation
    elapsed = (times - t0).sec
    n_slots = int(round(elapsed[-1] / cadence)) + 1

    # [t0, t0 + cadence, t0 + cadence * 2, ... + t0 + cadence * n_slots]
    grid_times = t0 + TimeDelta(np.arange(n_slots) * cadence, format='sec')

    # Corrects time indices to match the cadence
    # For example, [0, 45, 90, 135, 215] -> [0, 1, 2, 3, 5] with 45s cadence => index 4 is missing
    slot_indices = np.rint(elapsed / cadence).astype(int)

    # Map velocities to their corresponding time, with NaN to fill any non-matches
    grid_velocities = np.full(n_slots, np.nan, dtype=np.float32)
    grid_velocities[slot_indices] = velocities

    return grid_times, grid_velocities



def _read_velocity_sample(fits_file: Path, x: int, y: int) -> tuple[str, float]:
    """
    Read (t_rec, v(x, y)) from a single FITS file, where v(x, y) is velocity at (x, y) in m/s.
    Raises on malformed input.
    """
    with fits.open(fits_file) as hdul:
        # Use hdul.info() for more information on the structure of the header data unit list
        # In summary, index 0 is type PrimaryHDU, and index 1 is type CompImageHDU (compressed image).
        header = hdul[1].header
        data = hdul[1].data
        if not (0 <= y < data.shape[0] and 0 <= x < data.shape[1]):
            raise ValueError(f"({x}, {y}) out of bounds for shape {data.shape}")
        return _clean_t_rec(header['T_REC']), float(data[y, x])


def _clean_t_rec(t_rec: str) -> str:
    """
    Sanitizes T_REC from header data: YYYY.MM.DD_hh:mm:ss.sss_TAI -> YYYY-MM-DDThh:mm:ss.sss
    for easy conversion via astropy.time.Time.
    """
    return t_rec.removesuffix('_TAI').replace('.', '-', 2).replace('_', 'T')