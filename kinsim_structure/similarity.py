"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import pandas as pd
from scipy.spatial import distance


def calculate_similarity(fingerprint1, fingerprint2, measure):
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

    if measure == measures[0]:
        # Calculate inverse of the translated and scaled Manhattan distance
        return 1 / (1 + 1 / len(fingerprint1) * distance.cityblock(fingerprint1, fingerprint2))
    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')
