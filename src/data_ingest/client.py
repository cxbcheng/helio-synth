import os
import drms
import numpy as np
from dotenv import load_dotenv

def get_drms_client(email: str | None = None) -> drms.Client:
    """
    Factory function to initialize and validate a DRMS Client.
    """
    if not email:
        load_dotenv()
        email = os.environ.get('JSOC_EMAIL')
    if not email:
        raise ValueError("JSOC Email is required. Provide it as an argument or set JSOC_EMAIL in .env.")

    client = drms.Client(email=email)
    return client

def extract_velocity_timeseries(
    start_time: str,
    end_time: str,
    x: float,
    y: float,
    download_dir: str,
    email: str | None = None,
) -> np.ndarray:
    """
    TODO: documentation
    :param start_time:
    :param end_time:
    :param x:
    :param y:
    :param download_dir:
    :param email:
    :return:
    """
    client = get_drms_client(email)
    qstr = f"hmi.v_45s[{start_time}-{end_time}]"

    print(f"Submitting export request for {qstr} at ({x}, {y})...")
    # query = client.query(qstr, key="OBS_VR")
    # print(query.head)

    # FIXME: added record limit n = 10
    export_request = client.export(qstr, protocol='fits', n=10)
    print("Waiting for JSOC servers...")
    print(export_request.request_url)
    export_request.wait()

    print(export_request.data)


# FIXME: Remove tests
# Note how long these export requests take! Make sure to implement cutouts so that we don't try to
# extract the entire 4096x4096 image, even over the course of a day (which is about 3840 images).
extract_velocity_timeseries(
    start_time='2020-01-01T00:00:00Z',
    end_time='2020-01-01T00:59:59Z',
    x=0,
    y=0,
    download_dir='./',
)