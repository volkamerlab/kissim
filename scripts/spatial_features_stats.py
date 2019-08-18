import multiprocessing
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import SpatialFeatures


def get_spatial_features(klifs_metadata_entry):


    space = SpatialFeatures()
    ml = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
    space.from_molecule(ml.molecule)

    space.features['metadata_index'] = klifs_metadata_entry['index']
    space.features['kinase'] = klifs_metadata_entry.kinase
    space.features['pdb_id'] = klifs_metadata_entry.pdb_id
    space.features['chain'] = klifs_metadata_entry.chain
    space.features['alternate_model'] = klifs_metadata_entry.alternate_model
    space.features['qualityscore'] = klifs_metadata_entry.qualityscore
    space.features['resolution'] = klifs_metadata_entry.resolution

    return space.features


if __name__ == "__main__":

    path_to_data = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
    path_to_kinsim = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'
    path_to_results = path_to_kinsim / 'results'

    metadata_path = path_to_data / 'preprocessed' / 'klifs_metadata_preprocessed.csv'

    klifs_metadata = pd.read_csv(metadata_path)

    # Number of CPUs on machine
    num_cores = multiprocessing.cpu_count() - 1

    # Number of partitions to split DataFrame
    num_partitions = num_cores

    entry_list = [j for i, j in klifs_metadata.iterrows()]
    print(len(entry_list))

    # Create pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_cores)

    # Apply function to each chunk in list
    space_list = pool.map(get_spatial_features, entry_list)
    print(len(space_list))

    pool.close()
    pool.join()

    all_distances = pd.concat(space_list)
    print(len(all_distances))

    with open(path_to_results / 'postprocessing' / 'distances_all.p', 'wb') as f:
        pickle.dump(all_distances, f)
