import numpy as np
from PIL import Image
from matplotlib import colormaps


def remove_low_order_trend(data: np.ndarray, disk_mask: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Fits and subtracts a low-order 2D polynomial (orbital offset + rotational
    gradient are both smooth, low-spatial-frequency) over on-disk pixels only.
    TODO: documentation + idk how this works ngl
    """
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    yy_n = (yy - data.shape[0] / 2) / (data.shape[0] / 2)
    xx_n = (xx - data.shape[1] / 2) / (data.shape[1] / 2)

    terms = [np.ones_like(xx_n[disk_mask])]
    for total_deg in range(1, order + 1):
        for i in range(total_deg + 1):
            terms.append((xx_n[disk_mask] ** (total_deg - i)) * (yy_n[disk_mask] ** i))
    A = np.stack(terms, axis=1)

    coeffs, *_ = np.linalg.lstsq(A, data[disk_mask], rcond=None)

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
        out_size: int | None = None
) -> Image.Image:
    """
    Renders a (cleaned) velocity field as an RGBA image using a diverging
    colormap (red: toward observer, blue: away), transparent off-disk.
    """
    normalized = np.clip((velocity - v_min) / (v_max - v_min), v_min, v_max)
    rgba = colormaps['RdBu_r'](normalized, bytes=True)
    rgba[~disk_mask] = [0, 0, 0, 0]

    img = Image.fromarray(rgba, 'RGBA')
    return img.resize([out_size, out_size]) if out_size is not None else img
