from pathlib import Path

DATA_DIR = Path('data')
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Preprocessed data to remove the need of downloading JSOC data:
# for convenience in scripts and testing.
EXAMPLE_DATA_DIR = DATA_DIR / 'examples'