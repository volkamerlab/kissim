"""
generate_similarity_matrix.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Generate the similarity matrix for full KLIFS dataset.
"""

import logging

import datetime
import itertools
import multiprocessing
from pathlib import Path
import pickle

import pandas as pd

from kinsim_structure.similarity import get_fingerprint_type1_similarity, get_fingerprint_type2_similarity

PATH_TO_DATA = Path('/') / 'home' / 'dominique' / 'Documents' / 'data' / 'kinsim' / '20190724_full'
PATH_TO_KINSIM = Path('/') / 'home' / 'dominique' / 'Documents' / 'projects' / 'kinsim_structure'

# Set file and console logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_TO_KINSIM / 'results' / 'similarity' / 'generate_similarity_matrix.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def load_fingerprints(path_to_fingerprints):
    """
    Load fingerprints from file and remove fingerprints that are empty or that have empty features.

    Parameters
    ----------
    path_to_fingerprints : pathlib.Path or str
        Path to file containing list of fingerprints.

    Returns
    -------
    list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.
    """

    path_to_fingerprints = Path(path_to_fingerprints)

    # Load fingerprints
    with open(path_to_fingerprints, 'rb') as f:
        fingerprints = pickle.load(f)
    logger.info(f'Number of fingerprints: {len(fingerprints)}')

    # Remove empty fingerprints
    fingerprints = [i for i in fingerprints if i is not None]
    logger.info(f'Number of non-empty fingerprints: {len(fingerprints)}')
    logger.info(f'Empty fingerprints: {", ".join([i.molecule_code for i in fingerprints if i is None])}')

    # Remove fingerprints with empty features
    fingerprints = [i for i in fingerprints if i.fingerprint_type1 is not None]
    logger.info(f'Number of fingerprint with non-empty features: {len(fingerprints)}')
    logger.info(f'Empty fingerprint features: '
                f'{", ".join([i.molecule_code for i in fingerprints if i.fingerprint_type1 is None])}')

    return fingerprints


def get_pairs(fingerprints):
    """
    Get all pairwise combinations of fingerprints from list of fingerprints.

    Parameters
    ----------
    fingerprints : list of kinsim_structure.encoding.Fingerprint
        List of fingerprints.

    Returns
    -------
    List of 2-element list of fingerprints
        List of fingerprint pairs.
    """

    # Get fingerprint pairs
    pairs = []
    for i, j in itertools.combinations(fingerprints, 2):
        pairs.append([i, j])

    logger.info(f'Number of pairs: {len(pairs)}')

    return pairs


def get_pairwise_similarities(similarity_function, pairs):

    logger.info(f'{similarity_function.__name__}')

    # Number of CPUs on machine
    num_cores = multiprocessing.cpu_count() - 1
    logger.info(f'Number of cores used: {num_cores}')

    # Create pool with `num_processes` processes
    pool = multiprocessing.Pool(processes=num_cores)

    # Apply function to each chunk in list
    score_list = pool.map(similarity_function, pairs)

    # Close and join pool
    pool.close()
    pool.join()

    # Convert list to pandas.DataFrame
    scores_df = pd.DataFrame(
        score_list,
        columns='molecule1 molecule2 score score_physchem score_spatial coverage coverage_physchem coverage_spatial'.split()
    )

    logger.info(f'Number of scores (list): {len(score_list)}')
    logger.info(f'Number of scores (pandas.DataFrame): {len(scores_df)}')

    return scores_df


def main():

    # Get start time of script
    start = datetime.datetime.now()

    # Get fingerprint pairs
    fingerprints = load_fingerprints(PATH_TO_KINSIM / 'results' / 'fingerprints' / 'fingerprints.p')
    pairs = get_pairs(fingerprints)

    # Calculate pairwise similarity
    scores_fingerprint_type1 = get_pairwise_similarities(get_fingerprint_type1_similarity, pairs)
    scores_fingerprint_type2 = get_pairwise_similarities(get_fingerprint_type2_similarity, pairs)

    # Save similarities
    path_to_similarities = PATH_TO_KINSIM / 'results' / 'similarity'
    path_to_similarities.mkdir(parents=True, exist_ok=True)
    scores_fingerprint_type1.to_csv(path_to_similarities / 'scores_allxall_fingerprint_type1.csv')
    scores_fingerprint_type2.to_csv(path_to_similarities / 'scores_allxall_fingerprint_type2.csv')

    # Get end time of script
    end = datetime.datetime.now()

    logger.info(start)
    logger.info(end)


if __name__ == "__main__":
    main()
