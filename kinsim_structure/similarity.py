"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import numpy as np
import pandas as pd
from scipy.spatial import distance


def get_physchem_distances_similarity(fingerprint1, fingerprint2, physchem_weight=1, spatial_weight=1):



    physchem_score = None
    spatial_score = None

    score = physchem_weight * physchem_score + spatial_weight * spatial_score

    return score


def get_physchem_moments_similarity(fingerprint1, fingerprint2, physchem_weight=1, spatial_weight=1):
    pass


def calculate_similarity(fingerprint1, fingerprint2, measure='modified_manhattan'):
    """
    Calculate the similarity between two fingerprints based on a similarity measure.
    Parameters
    ----------
    fingerprint1 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    fingerprint2 : 1D array-like or pandas.DataFrame
        Fingerprint for molecule.
    measure : str
        Similarity measurement method:
         - modified_manhattan (inverse of the translated and scaled Manhattan distance)
    Returns
    -------
    float
        Similarity value.
    """

    measures = ['modified_manhattan']

    # Convert DataFrame into 1D array
    if isinstance(fingerprint1, pd.DataFrame):
        fingerprint1 = fingerprint1.values.flatten()
    if isinstance(fingerprint2, pd.DataFrame):
        fingerprint2 = fingerprint2.values.flatten()

    if len(fingerprint1) != len(fingerprint2):
        raise ValueError(f'Input fingerprints must be of same length.')

    print(type(fingerprint1))

    # Merge both fingerprints to array in order to remove positions with nan values
    fingerprints = np.array(
        [
            fingerprint1,
            fingerprint2
        ]
    ).transpose()

    # Remove nan positions (shall not be compared)
    fingerprints_reduced = fingerprints[
        ~np.isnan(fingerprints).any(axis=1)
    ]

    if measure == measures[0]:

        # Calculate inverse of the translated and scaled Manhattan distance
        fp1 = fingerprints_reduced[:, 0]
        fp2 = fingerprints_reduced[:, 1]

        return 1 / (1 + 1 / len(fp1) * distance.cityblock(fp1, fp2))

    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')
