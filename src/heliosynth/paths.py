from pathlib import Path

DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
TIMESERIES_DATA_DIR = PROCESSED_DATA_DIR / 'timeseries'
AUDIO_DATA_DIR = PROCESSED_DATA_DIR / 'audio'
IMAGES_DATA_DIR = PROCESSED_DATA_DIR / 'images'

# Preprocessed data to remove the need of downloading JSOC data:
# for convenience in scripts and testing.
EXAMPLE_DIR = Path('examples')
EXAMPLE_DATA_DIR = EXAMPLE_DIR / 'data'
EXAMPLE_DOWNLOAD_DIR = EXAMPLE_DIR / 'downloads'