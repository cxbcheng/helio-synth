import numpy as np
from astropy.time import Time, TimeDelta

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