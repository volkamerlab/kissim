"""
This script generates all fingerprints for a given dataset.
"""

import datetime
import multiprocessing
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader, PdbChainLoader
from kinsim_structure.encoding import Fingerprint


def get_fingerprint(klifs_metadata_entry):

    try:
        klifs_molecule_loader = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
        pdb_chain_loader = PdbChainLoader(klifs_metadata_entry=klifs_metadata_entry)

        molecule = klifs_molecule_loader.molecule
        chain = pdb_chain_loader.chain

        fp = Fingerprint()
        fp.from_molecule(molecule, chain)

        fp.features['metadata_index'] = klifs_metadata_entry.name
        fp.features['molecule_code'] = molecule.code

        return fp

    except:

        with open(path_to_results / 'fingerprints' / 'fingerprints_error_entries.txt', 'a+') as f_errors:
            error_message = f'{klifs_metadata_entry.species.upper()}/' \
                f'{klifs_metadata_entry.kinase}_' \
                f'{klifs_metadata_entry.pdb_id}_' \
                f'chain{klifs_metadata_entry.chain}_' \
                f'alt{klifs_metadata_entry.alternate_model}\n'
            f_errors.write(
                error_message
            )


if __name__ == "__main__":

    # Load IO paths
    path_to_data = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
    path_to_kinsim = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'
    path_to_results = path_to_kinsim / 'results'

    metadata_path = path_to_kinsim / 'data' / 'postprocessed' / 'klifs_metadata_postprocessed.csv'

    # Load metadata
    klifs_metadata = pd.read_csv(metadata_path)

    # Number of CPUs on machine
    num_cores = multiprocessing.cpu_count() - 1

    # Number of partitions to split DataFrame
    num_partitions = num_cores

    entry_list = [j for i, j in klifs_metadata.iterrows()]

    # Create pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_cores)

    start = datetime.datetime.now()

    # Apply function to each chunk in list
    fingerprints_list = pool.map(get_fingerprint, entry_list)

    pool.close()
    pool.join()

    end = datetime.datetime.now()
    print(start)
    print(end)

    # Save fingerprints
    (path_to_results / 'fingerprints').mkdir(parents=True, exist_ok=True)

    with open(path_to_results / 'fingerprints' / 'fingerprints_parallelized.p', 'wb') as f:
        pickle.dump(fingerprints_list, f)

    fingerprints = [i.features for i in fingerprints_list if i]
    fingerprints_df = pd.concat(fingerprints)

    # Save fingerprints
    fingerprints_df.to_csv(path_to_results / 'fingerprints' / 'fingerprints_parallelized.csv')
