from pathlib import Path

import numpy as np


def extract_velocity_timeseries(data_dir: str | Path, x: float, y: float) -> np.ndarray:
    """
    Parses local FITS files in the provided directory and extracts a velocity timeseries
    for a coordinate (x, y) on the solar disk (TODO: units?).
    :param data_dir:
    :return: Velocity time series extracted from the FITS files.
    """
    target_dir = Path(data_dir)
    raise NotImplementedError()