from pathlib import Path

import numpy as np
from PIL import Image
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.time import Time, TimeDelta

from heliosynth.data_ingest.utils import fits_name_to_time
from heliosynth.paths import RAW_DATA_DIR
from heliosynth.processing.imaging import detrend_disk_surface, render_dopplergram
from heliosynth.time_utils import require_tai


def extract_velocity_timeseries(
        data_dir: str | Path, x: int, y: int, cadence: int,
        start_time: Time | None = None,
        end_time: Time | None = None,
) -> tuple[Time, np.ndarray]:
    """
    Parses local FITS files in the provided directory and extracts a velocity timeseries
    for a pixel coordinate (x, y) on the solar disk image relative to the scale specified
    in the FITS file. Full resolution is 4096x4096; hence, valid positions include
    x, y \in [0, int(4096*scale)-1].
    If `start_index` or `end_index` specified, a subset of the files from the directory
    will be processed. Otherwise, all the files will be extracted.
    :param data_dir: Directory containing FITS files
    :param x: x pixel position
    :param y: y pixel position
    :param cadence: The expected time step or sampling interval (seconds) between any two
        data points. The time of the image is cross-validated with the cadence to mark
        any missing data with NaNs.
    :param start_time: If given, only files at or after this time are
        processed -- filtered from filenames before any FITS file is opened.
    :param end_time: If given, only files at or before this time are processed.
    :return: (times, velocities). A uniform-cadence timeseries. Velocity unit: m/s.
        velocities[i] is NaN whenever no valid sample was found for that time slot.
    :raises ValueError: if (x, y) is out of bounds, or no valid samples are found.
    """
    target_dir = Path(data_dir)
    fits_files = get_fits_files(start_time, end_time, target_dir)

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


def get_fits_files(
        start_time: Time | None = None,
        end_time: Time | None = None,
        dataset_dir: Path = RAW_DATA_DIR
) -> list[Path]:
    """
    Gets a list of the FITS files directly under a directory.
    :param start_time: If given, only files at or after this time are processed.
    :param end_time: If given, only files at or before this time are processed.
    :param dataset_dir: Directory containing FITS files.
    """
    fits_files = sorted(dataset_dir.glob('*.fits'))

    if start_time is not None or end_time is not None:
        fits_files = [
            f for f in fits_files
            if (start_time is None or fits_name_to_time(f.name) >= start_time)
               and (end_time is None or fits_name_to_time(f.name) <= end_time)
        ]
    return fits_files


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
    require_tai(times)
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


def load_dopplergram(fits_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the Dopplergram from the FITS file.
    Returns (velocity, disk_mask) -- the 2D Doppler velocity data and a boolean on-disk mask,
        intended to indicate where NaN/Inf values exist (due to being missing or off-disk).
    """
    with fits.open(fits_file) as hdul:
        velocity = hdul[1].data.astype(np.float32)
        disk_mask = np.isfinite(velocity)

    return velocity, disk_mask


def extract_solar_image(fits_file: Path, out_size: int | None = None,
                        v_min: float = -3000, v_max: float = 3000,
                        detrend_order: int | None = 2) -> Image.Image:
    """
    Convenience wrapper to load, clean, and render a single FITS file.
    Returns a PIL Image representing a diverging color colormap of the solar image.
    For extended documentation, see `load_dopplergram`, `detrend_disk_surface`,
    and `render_dopplergram`.

    The velocity values `v_min` and `v_max` should be set to the minimum and maximum
    of the signal over the sampling duration. These parameters are used to set the
    red and blue cutoffs for the colormap.

    If `detrend_order` is set to None, then the data is not detrended
    and a colormap of the raw Dopplergram image is returned.

    :param fits_file: Path to FITS file
    :param out_size: Size of output image
    :param v_min: Minimum velocity value (in m/s)
    :param v_max: Maximum velocity value (in m/s)
    :param detrend_order: Polynomial order used to detrend the surface.
        If set to `None`, the data is not detrended.
    """
    velocity, disk_mask = load_dopplergram(fits_file)
    if detrend_order is not None:
        velocity = detrend_disk_surface(velocity, disk_mask, detrend_order)
    return render_dopplergram(velocity, disk_mask, v_min=v_min, v_max=v_max, out_size=out_size)