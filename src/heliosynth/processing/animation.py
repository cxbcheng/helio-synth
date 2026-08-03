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
