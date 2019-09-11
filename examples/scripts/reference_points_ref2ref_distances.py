import multiprocessing
from pathlib import Path
import pickle

import pandas as pd
import scipy

from kinsim_structure.auxiliary import KlifsMoleculeLoader
from kinsim_structure.encoding import SpatialFeatures


def get_ref2ref_distances(klifs_metadata_entry):

    ml = KlifsMoleculeLoader(klifs_metadata_entry=klifs_metadata_entry)
    spatial = SpatialFeatures()
    spatial.from_molecule(ml.molecule)

    try:
        distances = scipy.linalg.norm(
            spatial.reference_points.apply(lambda x: x - spatial.reference_points.centroid),
            axis=0
        )

        return distances

    except AttributeError:
        print(spatial.reference_points)

        return ([None, None, None, None])


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
    result_list = pool.map(get_ref2ref_distances, entry_list)
    print(len(result_list))

    pool.close()
    pool.join()

    with open(path_to_results / 'ref2ref_distances_list.p', 'wb') as f:
        pickle.dump(result_list, f)

    result = pd.DataFrame(
        result_list,
        columns='centroid_to_centroid, centroid_to_hinge centroid_to_dfg centroid_to_front'.split()
    )
    result.reset_index(inplace=True)

    with open(path_to_results / 'ref2ref_distances.p', 'wb') as f:
        pickle.dump(result, f)

    klifs_metadata = klifs_metadata.copy()
    klifs_metadata.reset_index(inplace=True)

    result_df = pd.concat([klifs_metadata, result], axis=1)

    result_df.to_csv(path_to_results / 'ref2ref_distances.p')
