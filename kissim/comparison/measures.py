"""
kissim.comparison.measures

Defines the distance measures.
"""

import numpy as np
from scipy.spatial import distance


def scaled_euclidean_distance(vector1, vector2):
    """
    Calculate scaled Euclidean distance between two value lists of same length.

    Parameters
    ----------
    vector1 : np.ndarray
        Value list (same length as vector2).
    vector2 : np.ndarray
        Value list (same length as vector1).

    Returns
    -------
    float
        Scaled Euclidean distance between two value lists.
    """

    if len(vector1) != len(vector2):
        raise ValueError(f"Distance calculation failed: Values lists are not of same length.")
    elif len(vector1) == 0:
        return np.nan
    else:
        d = 1 / len(vector1) * distance.euclidean(vector1, vector2)
        return d


def scaled_cityblock_distance(vector1, vector2):
    """
    Calculate scaled cityblock distance between two value lists of same length.

    Parameters
    ----------
    vector1 : np.ndarray
        Value list (same length as vector2).
    vector2 : np.ndarray
        Value list (same length as vector1).

    Returns
    -------
    float
        Scaled cityblock distance between two value lists.
    """

    if len(vector1) != len(vector2):
        raise ValueError(f"Distance calculation failed: Values lists are not of same length.")
    elif len(vector1) == 0:
        return np.nan
    else:
        d = 1 / len(vector1) * distance.cityblock(vector1, vector2)
        return d
