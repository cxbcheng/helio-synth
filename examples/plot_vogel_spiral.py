"""
Vogel's spiral is a discretization of a Fermat spiral defined by

    rᵢ = R√(i/N),   θᵢ = iγ

where γ = π(3 − √5) is the golden angle, and i ∈ {1, …, N}.

Vogel's spiral has the property that it is asymptotically uniformly dense.
Since γ is irrational, α = γ/(2π) is irrational. By Weyl's
Equidistribution Theorem, the fractional parts of the sequence {nα},
for all nonzero integers n, are uniformly distributed modulo 1.
Thus, the angular coordinate is uniformly dense in [0, 2π).

By definition of rᵢ,

    πrᵢ² = πR²(i/N),

so the density is uniform with respect to area over the disk.

A uniformly dense surface is not necessarily optimally spaced -- this
becomes a spatial sampling design problem. For the scope of this project,
however, it is a simple and effective solution to choosing N samples to
extract data from (in order not to store every single pixel).
"""
from matplotlib import pyplot as plt

from heliosynth.sampling.vogel import construct_vogel_spiral


def main():
    # Change parameters here
    radius = 256
    n_points = 4000

    # By default, centered at (0, 0), but you can change that parameter here with center=(x,y)
    points = construct_vogel_spiral(
        n_points=n_points,
        radius=radius,
        snap_to_nearest_integer=True,
        include_center=True)

    # Plot the discretized spiral
    plt.figure(figsize=[10, 10])
    plt.title(f"Vogel's spiral with {n_points:,} points")
    x, y = points[:, 0][::1], points[:, 1][::1]
    plt.scatter(x, y, s=1, color='black')
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    plt.show()


if __name__ == "__main__":
    main()
