"""
Unit and regression test for kinsim_structure.similarity functions.
"""

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.similarity import calculate_similarity


@pytest.mark.parametrize('fingerprint1, fingerprint2, measure, score, coverage', [
    (
        pd.DataFrame([[1, 2], [3, 0]]),
        pd.DataFrame([[1, 1], [1, 1]]),
        'ballester',
        0.5,
        1.0
    ),
    (
        pd.DataFrame([[1, 2], [3, None]]),
        pd.DataFrame([[1, 1], [1, None]]),
        'ballester',
        0.5,
        0.75
    ),
    (
        pd.DataFrame([[1, 2], [3, np.nan]]),
        pd.DataFrame([[1, 1], [1, np.nan]]),
        'ballester',
        0.5,
        0.75
    )
])
def test_calculate_similarity(fingerprint1, fingerprint2, measure, score, coverage):
    """
    Test pairwise fingerprint similarity (similarity score and bit coverage) calculation given a defined measure.

    Parameters
    ----------
    fingerprint1 : pandas.DataFrame (or 1D array-like)
        Fingerprint for molecule.
    fingerprint2 : pandas.DataFrame (or 1D array-like)
        Fingerprint for molecule.
    measure : str
        Similarity measurement method:
         - ballester (inverse of the translated and scaled Manhattan distance)
    score : float
        Similarity score.
    coverage : float
         Coverage (ratio of bits used for similarity score).
    """

    assert calculate_similarity(fingerprint1, fingerprint2, measure='ballester') == (score, coverage)
