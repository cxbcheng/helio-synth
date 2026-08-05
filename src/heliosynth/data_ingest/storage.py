from pathlib import Path

import numpy as np
import zarr
from astropy.time import Time, TimeDelta
from numcodecs import Blosc

from heliosynth.time_utils import require_tai


def save_timeseries_npz(path, times: Time, velocities: np.ndarray, cadence: float) -> None:
    """
    Saves a timeseries npz file without need for pickling by
    encoding astropy.time.Time as (t0_jd1, t0_jd2, scale, cadence).
    """
    require_tai(times)
    np.savez(
        path,
        t0_jd1=times.jd1[0], t0_jd2=times.jd2[0],
        scale=times.scale,
        cadence=cadence,
        velocities=velocities,
    )


def load_timeseries_npz(path) -> tuple[Time, np.ndarray]:
    """
    Loads a timeseries npz file and decodes (t0_jd1, t0_jd2, scale, cadence) into an
    astropy.time.Time object.
    :return: (times, velocities)
    """
    with np.load(path) as f:
        t0 = Time(float(f['t0_jd1']), float(f['t0_jd2']), format='jd', scale=str(f['scale']))
        cadence = float(f['cadence'])
        n = f['velocities'].size
        times = t0 + TimeDelta(np.arange(n) * cadence, format='sec')
        return times, f['velocities']


def save_disk_velocity_zarr(
        path: Path,
        times: Time,
        velocities: np.ndarray,
        sample_points: np.ndarray,
        im_width: int,
) -> None:
    """
    Appends a batch of (times, velocities) to a zarr store of multipoint
    velocity samples, keyed by a fixed sample-point layout.

    Stores timestamps as jd1 and jd2 values.

    :param path: Zarr store directory (created if it doesn't exist).
    :param times: Per-frame observation times for this batch, TAI scale.
    :param velocities: (len(times), n_points) array; NaN where extraction
        failed for a given (time, point).
    :param sample_points: (n_points, 2) array of (x, y) centered-coordinate
        sample locations, as returned by construct_vogel_spiral -- stored
        BEFORE the row/col image-coordinate conversion, so it's reusable
        regardless of image orientation convention.
    :param im_width: Pixel width of the square image sample_points were
        computed against (needed to reinterpret them later).
    :raises ValueError: if times/velocities lengths mismatch, or an
        existing store's sample layout doesn't match this batch's --
        appending mismatched layouts would silently misalign columns.
    """
    require_tai(times)
    if len(times) != velocities.shape[0]:
        raise ValueError(f"times ({len(times)}) and velocities rows ({velocities.shape[0]}) mismatch")

    root = zarr.open_group(str(path), mode='a')
    n_points = velocities.shape[1]

    if 'velocities' in root:
        existing_points = root['sample_points'][:]
        if root.attrs['im_width'] != im_width or not np.array_equal(existing_points, sample_points):
            raise ValueError(
                f"Sample layout mismatch at {path}: existing store was built "
                f"with a different sample_points/im_width. Appending would "
                f"silently misalign columns -- use a separate store path for "
                f"a different sampling configuration."
            )
        vel_arr, jd1_arr, jd2_arr = root['velocities'], root['t_jd1'], root['t_jd2']
        n_old = vel_arr.shape[0]
        vel_arr.resize(n_old + len(times), n_points)
        vel_arr[n_old:] = velocities
        jd1_arr.resize(n_old + len(times))
        jd1_arr[n_old:] = times.jd1
        jd2_arr.resize(n_old + len(times))
        jd2_arr[n_old:] = times.jd2
    else:
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
        time_chunk = max(len(times), 1)
        vel_arr = root.create_dataset(
            'velocities', shape=velocities.shape, chunks=(time_chunk, n_points),
            dtype='f4', compressor=compressor,
        )
        vel_arr[:] = velocities  # type: ignore[index]
        for name, data in [('t_jd1', times.jd1), ('t_jd2', times.jd2)]:
            arr = root.create_dataset(name, shape=(len(times),), chunks=(time_chunk,), dtype='f8')
            arr[:] = data  # type: ignore[index]
        root.create_dataset('sample_points', data=sample_points, dtype='f8')
        root.attrs['im_width'] = im_width
        root.attrs['scale'] = times.scale
