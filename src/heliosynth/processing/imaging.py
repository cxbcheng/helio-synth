import numpy as np
from PIL import Image
from matplotlib import colormaps


def detrend_disk_surface(data: np.ndarray, disk_mask: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Fits and subtracts a 2D polynomial surface (the trend) from a disk.
    The polynomial surface is computed by fitting a 2D polynomial basis
    of degree 1 through degree `order` using least-squares.
    The fitted surface is then evaluated across the entire array and
    subtracted from the original data, returning a disk image which guarantees
    zero k^th order spatial moments, i.e.
    let R be the valid disk and k = `order`; then for each i \in {0, 1, ..., k},
        \sum_{(x, y)\in D} x^iy^{k-i}R(x,y)=0
    """
    if order < 0:
        raise ValueError("order must be a positive integer")

    # Normalization
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    yy_n = (yy - data.shape[0] / 2) / (data.shape[0] / 2)
    xx_n = (xx - data.shape[1] / 2) / (data.shape[1] / 2)

    # Build 2D polynomial basis on all masked pixels (x, y): {x^n, x^n-1 * y^1, ..., y^n}
    # This is used to build the N_pixels by N_terms design matrix
    terms = [np.ones_like(xx_n[disk_mask])]
    for total_deg in range(1, order + 1):
        for i in range(total_deg + 1):
            terms.append((xx_n[disk_mask] ** (total_deg - i)) * (yy_n[disk_mask] ** i))
    a = np.stack(terms, axis=1)
    coeffs, *_ = np.linalg.lstsq(a, data[disk_mask], rcond=None)

    # Reconstruct the trend
    full_terms = [np.ones_like(xx_n)]
    for total_deg in range(1, order + 1):
        for i in range(total_deg + 1):
            full_terms.append((xx_n ** (total_deg - i)) * (yy_n ** i))
    trend = np.tensordot(coeffs, np.stack(full_terms, axis=0), axes=1)

    return data - trend


def render_dopplergram(
        velocity: np.ndarray,
        disk_mask: np.ndarray,
        v_min: float = -3000,
        v_max: float = 3000,
        out_size: int | None = None,
        colormap: str = 'RdBu_r',
) -> Image.Image:
    """
    Renders a (cleaned) velocity field as an RGBA image using a diverging
    colormap (red: toward observer, blue: away), transparent off-disk.

    The optional parameter `colormap` chooses a colormap from
    the Matplotlib colormap registry:
    https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm._colormaps
    """
    normalized = np.clip((velocity - v_min) / (v_max - v_min), v_min, v_max)
    rgba = colormaps[colormap](normalized, bytes=True)
    rgba[~disk_mask] = [0, 0, 0, 0]

    img = Image.fromarray(rgba, 'RGBA')
    return img.resize([out_size, out_size]) if out_size is not None else img
