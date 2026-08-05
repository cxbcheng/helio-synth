"""
Downloads HMI Dopplergram FITS files from JSOC and stores them in RAW_DATA_DIR.

Edit the parameters in `main()` below (time range, pixel region, download scale)
to change what gets downloaded, then run this script directly:

    python download_hmi_data.py

Requires a JSOC-registered email address.

JSOC only serves export requests to registered email addresses. If this script
raises an error mentioning an unregistered or invalid email (or `get_drms_client`
raises `ValueError: JSOC Email is required...`), you need to register one first:

    1. Go to http://jsoc.stanford.edu/ajax/register_email.html
    2. Enter your email address and submit.
    3. Check your inbox for a confirmation link from JSOC and click it.
       Registration typically takes effect within a few minutes.

Without a confirmed registration, `client.export(...)` in `download_dopplergram_fits`
will fail or hang waiting on a request that JSOC silently refuses to process.

Once registered, set your email as an environment variable so you don't have to
pass it explicitly on every run. Create a file named `.env` in the project root
(if one doesn't already exist) containing:

    JSOC_EMAIL=your_registered_email@example.com

`get_drms_client` (see `heliosynth.data_ingest.client`) calls `load_dotenv()` and
reads this automatically. Alternatively, pass `email=...` directly to
`get_velocity_timeseries` / `download_dopplergram_fits` to bypass `.env` entirely (not recommended).

Do not commit `.env` to git (it should already be listed in `.gitignore`).
"""
from astropy.time import Time

from heliosynth.data_ingest.ingest import get_velocity_timeseries
from heliosynth.paths import RAW_DATA_DIR, EXAMPLE_DATA_DIR


def main():
    # Change parameters here to download different timeframes
    start_time = Time('2020-01-08 00:00:00', scale='tai')
    end_time = Time('2020-01-08 06:00:00', scale='tai')
    scale = 0.125  # 512x512 solar disk image (~20 kb per image)

    # Position to create a velocity time series from the downloaded files
    x = 256
    y = 256

    # Registered JSOC email address. Leave as None if it is already provided in the `.env` file.
    email = None

    """
    The data directories are set to their default values, so they do not need to be specified;
    nevertheless, they are explicitly provided for demonstration.
    
    Note that email is also an optional parameter -- the environmental variable `JSOC_EMAIL`
    will be used instead if left blank or None (recommended).
    """
    times, velocities = get_velocity_timeseries(
        start_time=start_time,
        end_time=end_time,
        x=x,
        y=y,
        scale=scale,
        download_missing=True,  # Downloads missing timestamps in the interval (if any)
        raw_data_dir=RAW_DATA_DIR,
        timeseries_data_dir=EXAMPLE_DATA_DIR,
        email=email)

    """
    Note that `get_velocity_timeseries` is a convenience wrapper which also handles
    extracting data from the FITS files and caching the NumPy array.
    Alternatively, you can choose to only download the data using
    `download_dopplergram_fits` with a subset of the same parameters.
    
    This is especially useful when your download gets interrupted and you want
    to recover the JSOC export request without resubmitting an identical one.
    """


if __name__ == "__main__":
    main()