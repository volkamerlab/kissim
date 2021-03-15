"""
Unit and regression test for the kissim.comparison.measures module.
"""

import pytest
import numpy as np
import pandas as pd

from kissim.comparison.utils import (
    format_weights,
    scaled_euclidean_distance,
    scaled_cityblock_distance,
)


@pytest.mark.parametrize(
    "feature_weights, feature_weights_formatted",
    [
        (None, np.array([0.0667] * 15)),
        (
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ),
    ],
)
def test_format_weights(feature_weights, feature_weights_formatted):
    """
    Test if feature weights are added correctly to feature distance DataFrame.

    Parameters
    ----------
    feature_weights : None or list of float
        Feature weights.
    feature_weights_formatted : list of float
        Formatted feature weights of length 15.
    """

    feature_weights_formatted_calculated = format_weights(feature_weights)

    assert np.isclose(
        np.std(feature_weights_formatted),
        np.std(feature_weights_formatted_calculated),
        rtol=1e-04,
    )


@pytest.mark.parametrize("feature_weights", [{"a": 0}, "bla"])
def test_format_weights_typeerror(feature_weights):
    """
    Test if wrong data type of input feature weights raises TypeError.
    """

    with pytest.raises(TypeError):
        format_weights(feature_weights)


@pytest.mark.parametrize(
    "feature_weights",
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0]],
)
def test_format_weights_valueerror(feature_weights):
    """
    Test if wrong data type of input feature weights raises TypeError.
    """

    with pytest.raises(ValueError):
        format_weights(feature_weights)


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

    score_calculated = scaled_euclidean_distance(vector1, vector2)

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
        scaled_euclidean_distance(vector1, vector2)


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

    score_calculated = scaled_cityblock_distance(vector1, vector2)

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
        scaled_cityblock_distance(vector1, vector2)
