"""
Unit and regression test for the kissim.comparison.measures module.
"""

import pytest
import numpy as np
import pandas as pd

from kissim.comparison import measures


@pytest.mark.parametrize(
    "vector1, vector2, distance",
    [
        ([], [], np.nan),
        ([0, 0], [4, 3], 2.5),
        (np.array([0, 0]), np.array([4, 3]), 2.5),
        (pd.Series([0, 0]), pd.Series([4, 3]), 2.5),
    ],
)
def test_scaled_euclidean_distance(vector1, vector2, distance):
    """
    Test Euclidean distance calculation.

    Parameters
    ----------
    vector1 : np.ndarray or list of pd.Series
        Value list (same length as vector2).
    vector2 : np.ndarray or list of pd.Series
        Value list (same length as vector1).
    distance : float
        Euclidean distance between two value lists.
    """

    score_calculated = measures.scaled_euclidean_distance(vector1, vector2)

    if not np.isnan(distance):
        assert np.isclose(score_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize(
    "vector1, vector2",
    [
        ([0, 0], [4, 3, 3]),
    ],
)
def test_scaled_euclidean_distance_raises(vector1, vector2):
    """
    Test if Euclidean distance calculation raises error if input values are of different
    length.
    """

    with pytest.raises(ValueError):
        measures.scaled_euclidean_distance(vector1, vector2)


@pytest.mark.parametrize(
    "vector1, vector2, distance",
    [
        ([], [], np.nan),
        ([0, 0], [4, 3], 3.5),
        (np.array([0, 0]), np.array([4, 3]), 3.5),
        (pd.Series([0, 0]), pd.Series([4, 3]), 3.5),
    ],
)
def test_scaled_cityblock_distance(vector1, vector2, distance):
    """
    Test Manhattan distance calculation.

    Parameters
    ----------
    vector1 : np.ndarray or list of pd.Series
        Value list (same length as vector2).
    vector2 : np.ndarray or list of pd.Series
        Value list (same length as vector1).
    distance : float
        Manhattan distance between two value lists.
    """

    score_calculated = measures.scaled_cityblock_distance(vector1, vector2)

    if not np.isnan(distance):
        assert np.isclose(score_calculated, distance, rtol=1e-04)


@pytest.mark.parametrize(
    "vector1, vector2",
    [
        ([0, 0], [4, 3, 3]),
    ],
)
def test_scaled_cityblock_distance_raises(vector1, vector2):
    """
    Test if Manhattan distance calculation raises error if input values are of different
    length.
    """

    with pytest.raises(ValueError):
        measures.scaled_cityblock_distance(vector1, vector2)
