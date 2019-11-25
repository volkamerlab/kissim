"""
klifs_metadata_filter.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Load and filter KLIFS metadata.
"""

import logging

from pathlib import Path
import pickle
import sys

sys.path.append('../..')
from kinsim_structure.preprocessing import KlifsMetadataLoader, KlifsMetadataFilter

PATH_SCRIPT = Path(__name__).parent
PATH_KLIFS_DOWNLOAD = Path('/home/dominique/Documents/data/kinsim/20191115_full/KLIFS_download')
FILENAME = 'klifs_metadata_filter'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_SCRIPT / f'{FILENAME}.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def main():

    # Load metadata
    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(
        PATH_KLIFS_DOWNLOAD / 'overview.csv',
        PATH_KLIFS_DOWNLOAD / 'KLIFS_export.csv'
    )

    # Filter metadata
    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.from_klifs_metadata(klifs_metadata_loader.data_essential, PATH_KLIFS_DOWNLOAD)

    # Save KlifsMetadataFilter object to disc
    with open(PATH_KLIFS_DOWNLOAD / f'{FILENAME}.p', 'wb') as f:
        pickle.dump(klifs_metadata_filter, f)


if __name__ == "__main__":
    main()
