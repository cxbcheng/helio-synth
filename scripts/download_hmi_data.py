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
`run_ingest` / `download_dopplergram_fits` to bypass `.env` entirely (not recommended).

Do not commit `.env` to git (it should already be listed in `.gitignore`).
"""
from heliosynth.data_ingest.ingest import run_ingest
from heliosynth.paths import TIMESERIES_DATA_DIR, RAW_DATA_DIR


def main():
    # Change parameters here to download different regions/timeframes
    start_time = "2020.01.01_00:00:00_TAI"
    end_time = "2020.01.08_00:00:00_TAI"
    x = 256
    y = 256
    scale = 0.125

    # Registered JSOC email address. Leave as None if it is already provided in the `.env` file.
    email = None

    """
    The data directories are set to their default values, so they do not need to be specified;
    nevertheless, they are explicitly provided for demonstration.
    
    Note that email is also an optional parameter -- the environmental variable `JSOC_EMAIL`
    will be used instead if left blank or None (recommended).
    """
    times, velocities = run_ingest(start_time=start_time, end_time=end_time, x=x, y=y, scale=scale,
                                   raw_data_dir=RAW_DATA_DIR,
                                   processed_data_dir=TIMESERIES_DATA_DIR,
                                   email=email)


if __name__ == "__main__":
    main()