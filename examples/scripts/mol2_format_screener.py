"""
mol2_format_screener.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Screen all KLIFS protein.mol2 files for irregular formats, i.e. underscored residues, non-standard residues, and
residues with duplicated atom names.
"""

import logging

from pathlib import Path
import pickle
import sys

sys.path.append('../..')
from kissim.preprocessing import KlifsMetadataLoader, Mol2FormatScreener

PATH_SCRIPT = Path(__name__).parent
PATH_KLIFS_DOWNLOAD = Path('/home/dominique/Documents/data/kinsim/20191115_full/KLIFS_download')
FILENAME = 'mol2_format_screener'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_KLIFS_DOWNLOAD.parent / f'{FILENAME}.log',
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
    
    # Screen protein.mol2 file for irregular formats
    klifs_metadata = klifs_metadata_loader.data_essential
    mol2_format_screener = Mol2FormatScreener()
    mol2_format_screener.from_metadata(klifs_metadata, PATH_KLIFS_DOWNLOAD)

    # Save Mol2FormatScreener object to disc
    with open(PATH_KLIFS_DOWNLOAD.parent / f'{FILENAME}.p', 'wb') as f:
        pickle.dump(mol2_format_screener, f)


if __name__ == "__main__":
    main()
