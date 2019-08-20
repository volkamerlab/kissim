"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial import distance

from kinsim_structure.encoding import FEATURE_NAMES

logger = logging.getLogger(__name__)


def get_physchem_distances_similarity(pair, measure='modified_manhattan', weight=None):
    """
    Get similarity score for fingerprint type1 (consisting of physicochemical and distance properties) based on a
    similarity measure (default modified Manhattan distance).
    Option to weight physicochemical and distance properties differently (default None).

    Parameters
    ----------
    pair : 2-element-list of kinsim_structure.encoding.Fingerprint
        bla
    measure : str
        Similarity measure name.
    weight : None or float [0,1]
        If weight is None (default), fingerprint similarity is calculated over all positions.
        If weight is given, similarity for physicochemical and distance fingerprint bits are calculated separately, and
        summed up with respect to assigned weight for the phsicochemical part and (1-weight) for the distance part.

    Returns
    -------
    List
        List of molecule names in pair and their score.
    """

    if not weight:

        score = calculate_similarity(
            pair[0].features.iloc[:, 0:12],
            pair[1].features.iloc[:, 0:12],
            measure=measure
        )

        return [
            pair[0].molecule_code,
            pair[1].molecule_code,
            score
        ]

    else:

        if 0 <= weight <= 1:

            print(pair[0].molecule_code, pair[1].molecule_code)

            physchem_score = calculate_similarity(
                pair[0].features.iloc[:, 0:8],
                pair[1].features.iloc[:, 0:8],
                measure=measure
            )
            spatial_score = calculate_similarity(
                pair[0].features.iloc[:, 9:12],
                pair[1].features.iloc[:, 9:12],
                measure=measure
            )

            score = weight * physchem_score + (1-weight) * spatial_score

        else:
            raise ValueError(f'Weight must be between 0 and 1. Given weight is: {weight}')

        return [
            pair[0].molecule_code,
            pair[1].molecule_code,
            score
        ]


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

    print(fingerprints_reduced)

    if measure == measures[0]:

        # Calculate inverse of the translated and scaled Manhattan distance
        fp1 = fingerprints_reduced[:, 0]
        fp2 = fingerprints_reduced[:, 1]

        try:

            score = 1 / (1 + 1 / len(fp1) * distance.cityblock(fp1, fp2))

        except ZeroDivisionError:

            score = None

        return score

    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')
