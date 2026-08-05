import numpy as np

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5))


def construct_vogel_spiral(
    n_points: int,
    radius: float,
    center: tuple[float, float] = (0, 0),
    snap_to_nearest_integer: bool = False,
    include_center: bool = True,
) -> np.ndarray:
    """
    Vogel's spiral is a discretization of a Fermat spiral defined by
        r_i = R\sqrt{i/N}, \theta_i=i\gamma
    where \gamma = \pi(3-\sqrt{5}) is the golden angle, and i = \{1, ..., N\}.

    Vogel's spiral has the property that it is asymptotically uniformly dense.
    For, since \gamma is irrational, we know by Weyl's Equidistribution Theorem
    that \alpha = \frac{\gamma}{2\pi} means that the sequence {n\g}
    for all nonzero integers n is uniformly distributed modulo 1, thus
    its angle is uniformly dense in [0, 2\pi).
    And by definition of r_i, we have \pi r_i^2 = \pi R^2/N, meaning
    radius is uniformly dense in [0, R].

    A uniformly dense surface is not necessarily optimally spaced, but for our
    use cases it is good enough.

    :param n_points: Number of points to return.
    :param radius: Radius of the spiral.
    :param center: Coordinates of the center of the spiral.
    :param snap_to_nearest_integer: Whether to round points to integer values.
    :param include_center: If set to `True`, the spiral is constructed using
        `n_points - 1` spiral points with indices i = \{1, ..., N-1\}, and
        the center is prepended so that the array contains `n_points`.
    :return: Positions of the N points on the spiral on the Cartesian plane.
        If `snap_to_nearest_integer` is True, the type of the array is set to int.
    """
    if n_points < 0:
        raise ValueError("n_points cannot be negative.")
    if include_center:
        n_points -= 1
    if n_points == 0:
        return np.empty((0, 2), dtype=np.float64)

    cx, cy = center
    i = np.arange(1, n_points + 1)
    r = radius * np.sqrt(i / n_points)
    theta = i * GOLDEN_ANGLE
    points = np.column_stack([cx + r * np.cos(theta), cy + r * np.sin(theta)])

    if include_center:
        points = np.vstack(([cx, cy], points))
    if snap_to_nearest_integer:
        points = np.rint(points).astype(np.int32)
    return points
