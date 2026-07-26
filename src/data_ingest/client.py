import os
from dotenv import load_dotenv
import drms

load_dotenv()
email_address = os.environ.get('JSOC_EMAIL')
out_dir = os.environ.get('DATA_DIR')

if not email_address:
    raise ValueError('Email address not specified. Set the environmental variable JSOC_EMAIL as your JSOC email address.')
if not out_dir or not os.path.exists(out_dir):
    raise ValueError('Data directory not specified or its path does not exist. Set the environmental variable DATA_DIR.')

client = drms.Client(email=email_address)