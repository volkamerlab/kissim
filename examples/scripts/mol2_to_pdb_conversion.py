import logging

from pathlib import Path
import sys

sys.path.append('../..')
from kinsim_structure.preprocessing import KlifsMetadataLoader, KlifsMetadataFilter, Mol2ToPdbConverter

PATH_TO_KLIFS_DOWNLOAD = Path('/home/dominique/Documents/data/kinsim/20190724_full/raw')
PATH_TO_SCRIPT = Path(__name__).parent

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_TO_SCRIPT / 'mol2_to_pdb_conversion.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def main():

    klifs_overview_file = PATH_TO_KLIFS_DOWNLOAD / 'KLIFS_download' / 'overview.csv'
    klifs_export_file = PATH_TO_KLIFS_DOWNLOAD / 'KLIFS_download' / 'KLIFS_export.csv'

    klifs_metadata_loader = KlifsMetadataLoader()
    klifs_metadata_loader.from_files(klifs_overview_file, klifs_export_file)

    klifs_metadata_filter = KlifsMetadataFilter()
    klifs_metadata_filter.from_klifs_metadata(klifs_metadata_loader.data_reduced, PATH_TO_KLIFS_DOWNLOAD)

    klifs_metadata = klifs_metadata_filter.filtered
    logger.info(f'Number of metadata entries: {len(klifs_metadata)}')

    converter = Mol2ToPdbConverter()
    converter.from_klifs_metadata(klifs_metadata, PATH_TO_KLIFS_DOWNLOAD)


if __name__ == "__main__":
    main()
