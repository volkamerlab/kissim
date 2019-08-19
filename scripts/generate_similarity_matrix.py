

import datetime
import itertools
import multiprocessing
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.similarity import get_physchem_distances_similarity


if __name__ == "__main__":

    # Load IO paths
    path_to_data = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
    path_to_kinsim = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'
    path_to_results = path_to_kinsim / 'results'

    # Load fingerprints
    with open(path_to_results / 'fingerprints' / 'fingerprints_parallelized.p', 'rb') as f:
        fingerprints = pickle.load(f)

    print(f'Number of fingerprints: {len(fingerprints)}')

    # Remove None entries in fingerprint list
    fingerprints = [i for i in fingerprints if i is not None]
    print(f'Number of non-empty fingerprints: {len(fingerprints)}')

    empty_features = [i.molecule_code for i in fingerprints if i.features is None]
    print(f'Empty fingerprints: {empty_features}')

    # Test #TODO
    fingerprints = fingerprints[:100000]

    # Number of CPUs on machine
    num_cores = multiprocessing.cpu_count() - 1

    # Number of partitions to split DataFrame
    num_partitions = num_cores

    pairs = []
    for i, j in itertools.combinations(fingerprints, 2):
        pairs.append([i, j])

    # Create pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_cores)

    start = datetime.datetime.now()

    # Apply function to each chunk in list
    score_list = pool.map(get_physchem_distances_similarity, pairs)

    pool.close()
    pool.join()

    end = datetime.datetime.now()
    print(start)
    print(end)

    # Save fingerprints
    (path_to_results / 'comparison').mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(
        score_list,
        columns='molecule1 molecule2 score'.split()
    )

    print(f'Number of scores: {len(scores_df)}')

    # Save fingerprints
    scores_df.to_csv(path_to_results / 'comparison' / 'scores_allxall_fingerprint_type1.csv')
