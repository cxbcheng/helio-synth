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

Additionally, you have the option below to change the colormap of
the rendered Dopplergrams. The default is 'RdBu_r', but you can choose any
from the list of registered colormaps from Matplotlib:

>>> from matplotlib import colormaps
>>> list(colormaps)
"""
from astropy.time import Time
from matplotlib import pyplot as plt

from heliosynth.constants import V_MIN, V_MAX
from heliosynth.data_ingest.extraction import extract_solar_image, read_dopplergram
from heliosynth.data_ingest.utils import get_fits_path
from heliosynth.paths import EXAMPLE_DATA_DIR


def main():
    # Parameter to change the displayed colormap
    colormap = 'RdBu_r'

    # Here is a pre-downloaded FITS file from `data/examples` for convenience
    time = Time('2020-01-01 00:00:00', scale='tai')
    fits_path = get_fits_path(dataset_dir=EXAMPLE_DATA_DIR, time=time)

    # Demo: the effect of different polynomial trend removals
    for trend_order in (None, 0, 2):
        # Choose velocity bounds for the colormap (we calculate this in `examples/print_velocity_stats.py`)
        v_min = V_MIN[trend_order]
        v_max = V_MAX[trend_order]

        # Load the Dopplergram as a PIL image
        img = extract_solar_image(read_dopplergram(fits_path), v_min=v_min, v_max=v_max, detrend_order=trend_order, colormap=colormap)

        # We can plot the PIL images easily
        plt.figure(figsize=(6, 6))
        plt.imshow(img)

        if trend_order is None:
            plt.title("Raw Dopplergram\n(No Trend Removal)")
        elif trend_order == 0:
            plt.title("Detrended Dopplergram\n (Trend Order = 0: Mean Velocity Removed)")
        else:
            plt.title("Detrended Dopplergram\n(Trend Order = 2: Large-Scale Velocity Removed)")

        plt.xlabel(time.iso)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()