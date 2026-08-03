import numpy as np
from astropy.time import Time, TimeDelta

from heliosynth.time_utils import require_tai


def get_animation_timestamps(
    start_time: Time,
    end_time: Time,
    fps: int,
    alpha: float,
) -> Time:
    """
    Returns a Time vector of the ideal timestamps needed to be able to render
    these at `fps` while still maintaining the playback speed `alpha`.
    Effectively, playback at `fps` frames per second advances through the
    observations at `alpha` seconds of solar time per second of animation.
    """
    require_tai(start_time)
    require_tai(end_time)
    step = alpha / fps
    n_frames = int((end_time - start_time).sec / step) + 1
    return start_time + TimeDelta(np.arange(n_frames) * step, format='sec')



def snap_to_available(times: Time, available: Time) -> tuple[Time, np.ndarray]:
    """
    Snaps each time in `times` to the nearest timestamp in `available`.
    Times not in the `available` range are clamped to the nearest endpoint.

    Assumes times and available are sorted ascending.

    Returns (snapped_times, indices), where snapped_times = available[indices].
    """
    require_tai(times)
    require_tai(available)
    if len(available) == 0:
        raise ValueError("available cannot be empty")

    # We note that if times[i] > available[j] for n values of j,
    # then times[i] is between available[n-1] and available[n]
    indices = np.searchsorted(available, times)
    indices = np.clip(indices, 1, len(available) - 1)

    # Choose the closer of the two: available[n-1] or available[n]
    floor, ceiling = indices - 1, indices
    snap_up = np.abs(available[ceiling] - times) <= np.abs(available[floor] - times)
    indices = np.where(snap_up, ceiling, floor)

    return available[indices], indices
