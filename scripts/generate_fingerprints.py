"""
This script generates all fingerprints for a given dataset.
"""

import logging

import datetime
import multiprocessing
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint

PATH_TO_DATA = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
PATH_TO_KINSIM = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'

# Set file and console logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_TO_KINSIM / 'results' / 'fingerprints' / 'generate_fingerprints.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def load_metadata(path_to_metadata):

    klifs_metadata = pd.read_csv(path_to_metadata)

    logger.info(f'Number of metadata entries: {len(klifs_metadata)}')

    return klifs_metadata


def get_fingerprint(klifs_metadata_entry):

    klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
    pdb_chain_loader = PdbChainLoader(klifs_metadata_entry=klifs_metadata_entry)

    molecule = klifs_molecule_loader.molecule
    chain = pdb_chain_loader.chain

    try:

        fp = Fingerprint()
        fp.from_molecule(molecule, chain)

        return fp

    except:

        error_message = f'{klifs_metadata_entry.species.upper()}/' \
            f'{klifs_metadata_entry.kinase}_' \
            f'{klifs_metadata_entry.pdb_id}_' \
            f'chain{klifs_metadata_entry.chain}_' \
            f'alt{klifs_metadata_entry.alternate_model}\n'

        logger.info(f'{error_message}')


def get_fingerprints(klifs_metadata):

    # Number of CPUs on machine
    num_cores = multiprocessing.cpu_count() - 1

    entry_list = [j for i, j in klifs_metadata.iterrows()]

    # Create pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_cores)

    # Apply function to each chunk in list
    fingerprints_list = pool.map(get_fingerprint, entry_list)

    pool.close()
    pool.join()

    return fingerprints_list


def main():

    # Get start time of script
    start = datetime.datetime.now()

    # Get metadata entries
    klifs_metadata = load_metadata(PATH_TO_KINSIM / 'data' / 'postprocessed' / 'klifs_metadata_postprocessed.csv')
    klifs_metadata = klifs_metadata[:100]  # TODO

    # Calculate fingerprints
    fingerprints_list = get_fingerprints(klifs_metadata)

    # Save fingerprints
    path_to_fingerprints = PATH_TO_KINSIM / 'results' / 'fingerprints'
    path_to_fingerprints.mkdir(parents=True, exist_ok=True)
    with open(path_to_fingerprints / 'fingerprints.p', 'wb') as f:
        pickle.dump(fingerprints_list, f)

    # Get end time of script
    end = datetime.datetime.now()

    logger.info(start)
    logger.info(end)


if __name__ == "__main__":
    main()
