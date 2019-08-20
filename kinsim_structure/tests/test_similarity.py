"""
Unit and regression test for kinsim_structure.similarity functions.
"""

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.encoding import Fingerprint, FEATURE_NAMES
from kinsim_structure.similarity import calculate_similarity
from kinsim_structure.similarity import get_physchem_distances_similarity


@pytest.mark.parametrize('fingerprint1, fingerprint2, measure, score, coverage', [
    (
        pd.DataFrame([[1, 2], [3, 0]]),
        pd.DataFrame([[1, 1], [1, 1]]),
        'modified_manhattan',
        0.5,
        1.0
    ),
    (
        pd.DataFrame([[1, 2], [3, None]]),
        pd.DataFrame([[1, 1], [1, None]]),
        'modified_manhattan',
        0.5,
        0.75
    ),
    (
        pd.DataFrame([[1, 2], [3, np.nan]]),
        pd.DataFrame([[1, 1], [1, np.nan]]),
        'modified_manhattan',
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
         - modified_manhattan (inverse of the translated and scaled Manhattan distance)
    score : float
        Similarity score.
    coverage : float
         Coverage (ratio of bits used for similarity score).
    """

    assert calculate_similarity(fingerprint1, fingerprint2, measure='modified_manhattan') == (score, coverage)


@pytest.mark.parametrize('fingerprint_df1, fingerprint_df2, measure, weight, score, coverage', [
    (
        pd.DataFrame(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            columns=FEATURE_NAMES
        ),
        pd.DataFrame(
            [
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            ],
            columns=FEATURE_NAMES
        ),
        'modified_manhattan',
        None,
        0.4,
        1.0
    ),
    (
        pd.DataFrame(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            columns=FEATURE_NAMES
        ),
        pd.DataFrame(
            [
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            ],
            columns=FEATURE_NAMES
        ),
        'modified_manhattan',
        0.1,
        0.49,
        (1.0, 1.0)
    )
])
def test_get_physchem_distances_similarity(fingerprint_df1, fingerprint_df2, measure, weight, score, coverage):

    # Set fingerprint 1
    fp1 = Fingerprint()
    fp1.molecule_code = 'molecule1'
    fp1.features = fingerprint_df1

    # Set fingerprint 2
    fp2 = Fingerprint()
    fp2.molecule_code = 'molecule2'
    fp2.features = fingerprint_df2

    pair = [fp1, fp2]

    result = get_physchem_distances_similarity(pair, measure, weight)

    assert result[0] == 'molecule1'
    assert result[1] == 'molecule2'
    assert result[2] == score
    assert result[3] == coverage
