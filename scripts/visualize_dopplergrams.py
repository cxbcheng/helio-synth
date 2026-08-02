"""
Demonstrates spatial detrending of an HMI Dopplergram.
Also serves as a tutorial implementation to extract Dopplergram data.
Data in the example directory has been provided for convenience.

Plots three Dopplergrams:

1. No detrending
   Displays the raw line-of-sight velocity field, containing the
   disk-averaged velocity, solar rotation, and smaller-scale solar
   velocity structures.

2. Trend order 0
   Removes the disk-averaged (constant) velocity, emphasizing the
   large-scale rotational Doppler pattern.

3. Trend order 2
   Removes a quadratic spatial trend, suppressing the dominant
   large-scale velocity field and revealing smaller-scale features,
   including granulation and p-mode oscillations.
"""

from matplotlib import pyplot as plt

from heliosynth.data_ingest.extraction import extract_solar_image
from heliosynth.paths import EXAMPLE_DATA_DIR


def main():
    # Here is a pre-downloaded FITS file for convenience
    fits_path = EXAMPLE_DATA_DIR / 'hmi.v_45s.20200101_000000_TAI.2.Dopplergram.fits'

    # Demo: the effect of different polynomial trend removals
    for trend_order in (None, 0, 2):
        # Load the Dopplergram as a PIL image
        img = extract_solar_image(fits_file=fits_path, detrend_order=trend_order)

        # We can plot the PIL images easily
        plt.figure(figsize=(6, 6))
        plt.imshow(img)

        if trend_order is None:
            plt.title("Raw Dopplergram\n(No Trend Removal)")
        elif trend_order == 0:
            plt.title("Detrended Dopplergram\n (Trend Order = 0: Mean Velocity Removed)")
        else:
            plt.title("Detrended Dopplergram\n(Trend Order = 2: Large-Scale Velocity Removed)")

        plt.xlabel("January 1st, 2020 (00:00:00)")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()