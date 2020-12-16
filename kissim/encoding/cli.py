"""
cli_encoding.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Execute encoding step.
"""

import argparse
import pickle
import logging
from pathlib import Path

from kissim.main import Encoding

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='Path to input (data) folder', required=True)
parser.add_argument('-o', '--output', type=str, help='Path to output (results) folder', required=True)
args = parser.parse_args()

# Parameters
PATH_DATA = Path(args.input)
PATH_RESULTS = Path(args.output)

PATH_KLIFS_DOWNLOAD = PATH_DATA / 'KLIFS_download'

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_RESULTS / 'cli_encoding.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


if __name__ == "__main__":

    with open(PATH_RESULTS / 'preprocessing' / 'klifs_metadata_filter.p', 'rb') as f:
        klifs_metadata_filter = pickle.load(f)

    encoding = Encoding()
    fingerprint_generator = encoding.execute(
        klifs_metadata_filter,
        PATH_KLIFS_DOWNLOAD,
	PATH_RESULTS
    )





