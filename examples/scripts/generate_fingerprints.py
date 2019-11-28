"""
generate_fingerprints.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Generate fingerprints.
"""

import logging

from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.append('../..')
from kinsim_structure.encoding import FingerprintGenerator

PATH_TO_KINSIM = Path('.') / '..' / '..'
PATH_TO_METADATA = PATH_TO_KINSIM / 'examples' / 'data' / 'postprocessed' / 'klifs_metadata_postprocessed.csv'
PATH_TO_RESULTS = PATH_TO_KINSIM / 'examples' / 'results' / 'fingerprints'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_TO_RESULTS / 'generate_fingerprints.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def main():

    # Get load
    klifs_metadata = pd.read_csv(PATH_TO_METADATA)
    logger.info(f'Number of metadata entries: {len(klifs_metadata)}')

    # Calculate fingerprints
    fingerprint_generator = FingerprintGenerator()
    fingerprint_generator.from_metadata(klifs_metadata[:20])

    # Save fingerprints
    with open(PATH_TO_RESULTS / 'fingerprints.p', 'wb') as f:
        pickle.dump(fingerprint_generator, f)


if __name__ == "__main__":
    main()
