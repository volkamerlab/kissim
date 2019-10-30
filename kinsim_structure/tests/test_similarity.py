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




@pytest.mark.parametrize('values1, values2, values_reduced, coverage', [
    ([0, 0, np.nan, 1], [4, 3, 1, np.nan], [[0, 0], [4, 3]], 0.5),
    ([0, 0, np.nan], [4, 3, np.nan], [[0, 0], [4, 3]], 0.6667),
    ([0, 0], [4, 3], [[0, 0], [4, 3]], 1.0)
])
def test_get_values_without_nan(values1, values2, values_reduced, coverage):

    values_reduced_calculated = _get_values_without_nan(values1, values2)

    assert np.isclose(values_reduced_calculated['coverage'], coverage, rtol=1e-04)
    assert np.isclose(values_reduced_calculated['values'], values_reduced, rtol=1e-04).all()


@pytest.mark.parametrize('values1, values2, distance', [
    ([0, 0], [4, 3], 2.5),
    (pd.Series([0, 0]), pd.Series([4, 3]), 2.5)
])
def test_euclidean_distance(values1, values2, distance):
    """
    Test Euclidean distance calculation.

    Parameters
    ----------
    values1 : list or pandas.Series
        Value list (same length as values2).
    values2 : list or pandas.Series
        Value list (same length as values1).
    distance : float
        Euclidean distance between two value lists.
    """

    score_calculated = _euclidean_distance(values1, values2)

    assert np.isclose(score_calculated, distance, rtol=1e-04)