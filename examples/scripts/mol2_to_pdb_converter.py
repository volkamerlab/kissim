"""
mol2_to_pdb_converter.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Convert mol2 files, which are linked to KLIFS metadata, to pdb files using PyMol.
"""

import logging

from pathlib import Path
import sys

sys.path.append('../..')
from kinsim_structure.preprocessing import KlifsMetadataLoader, Mol2ToPdbConverter

PATH_SCRIPT = Path(__name__).parent
PATH_KLIFS_DOWNLOAD = Path('/home/dominique/Documents/data/kinsim/20191115_full/KLIFS_download')
FILENAME = 'mol2_to_pdb_converter'

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

    # Convert mol2 to pdb files
    converter = Mol2ToPdbConverter()
    converter.from_klifs_metadata(klifs_metadata_loader.data_essential, PATH_KLIFS_DOWNLOAD)


if __name__ == "__main__":
    main()
