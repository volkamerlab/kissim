"""
mol2_klifs_to_pymol_converter.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Format KLIFS mol2 file to PyMol readable mol2 file.
"""

import logging

from pathlib import Path
import pickle
import sys

sys.path.append('../..')
from kissim.preprocessing import KlifsMetadataLoader, Mol2KlifsToPymolConverter

PATH_SCRIPT = Path(__name__).parent
PATH_KLIFS_DOWNLOAD = Path('/home/dominique/Documents/data/kinsim/20191115_full/KLIFS_download')
FILENAME = 'mol2_klifs_to_pymol_converter'

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
    mol2_klifs_to_pymol_converter = Mol2KlifsToPymolConverter()
    mol2_klifs_to_pymol_converter.from_metadata(klifs_metadata, PATH_KLIFS_DOWNLOAD)


if __name__ == "__main__":
    main()
