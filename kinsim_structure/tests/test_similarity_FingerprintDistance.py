"""
Unit and regression test for kinsim_structure.similarity functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.similarity import FingerprintDistance


@pytest.mark.parametrize('feature_type_weights', [
    ({'physicochemical': 0.1}),  # Features missing
    ({'physicochemical': 0.1, 'xxx': 0.1}),  # Unknown feature
    ({
        'physicochemical': 0.5,
        'distances': 0.5,
        'moments': 0.5
    }),  # Weights do not sum up to 1.0
])
def test_format_weight_per_feature_type_valueerror(feature_type_weights):
    """
    Test if incorrect input feature type weights raise ValueError.

    Parameters
    ----------
    feature_type_weights : dict of float
        Dictionary does not fulfill one or more of these conditions:
        Weights per feature which need to sum up to 1.0.
        Feature types to be set are: physicochemical, distances, and moments.
    """

    with pytest.raises(ValueError):
        fingerprint_distance = FingerprintDistance()
        fingerprint_distance._format_weight_per_feature_type(feature_type_weights)


@pytest.mark.parametrize('feature_type_weights', [
    ({
        'physicochemical': 'bla',
        'distances': 0.5,
        'moments': 0.5
    }),  # Weight value is not float
    ({
        'physicochemical': 1,
        'distances': 0,
        'moments': 0
    }),  # Weight value is not float
])
def test_format_weight_per_feature_type_typeerror(feature_type_weights):
    """
    Test if incorrect input feature type weights raise TypeError.

    Parameters
    ----------
    feature_type_weights : dict of float
        Feature types to be set are: physicochemical, distances, and moments.
        Set dictionary values to data type other than float.
    """

    with pytest.raises(TypeError):
        fingerprint_distance = FingerprintDistance()
        fingerprint_distance._format_weight_per_feature_type(feature_type_weights)


@pytest.mark.parametrize('feature_type_weights, feature_weights, weight_column_dtype, feature_name_column_dtype, shape', [
    (
        {
            'physicochemical': 0.0,
            'distances': 1.0,
            'moments': 0.0
        },
        {
            'size': 0.0,
            'hbd': 0.0,
            'hba': 0.0,
            'charge': 0.0,
            'aromatic': 0.0,
            'aliphatic': 0.0,
            'sco': 0.0,
            'exposure': 0.0,
            'distance_to_centroid': 0.25,
            'distance_to_hinge_region': 0.25,
            'distance_to_dfg_region': 0.25,
            'distance_to_front_pocket': 0.25,
            'moment1': 0.0,
            'moment2': 0.0,
            'moment3': 0.0
        },
        'float64',
        'object',
        (15, 2)
    )
])
def test_format_weight_per_feature_type(feature_type_weights, feature_weights, weight_column_dtype, feature_name_column_dtype, shape):
    """
    Test formatting of weights per feature type (weights need to be equally distributed between all features in feature
    type and transformed into a DataFrame).

    Parameters
    ----------
    feature_type_weights : dict of float (3 items) or None
        Weights per feature type which need to sum up to 1.0.
        Feature types to be set are: physicochemical, distances, and moments.
        Default feature weights (None) are set equally distributed to 1/3 (3 feature types in total).
    feature_weights : dict of float (15 items) or None
        Weights per feature which need to sum up to 1.0.
        Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure, distance_to_centroid,
        distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
        Default feature weights (None) are set equally distributed to 1/15 (15 feature in total).
    weight_column_dtype : str
        Data type of weight returned DataFrame column.
    feature_name_column_dtype : str
        Data type of feature_name returned DataFrame column.
    shape : tuple
        Dimension of returned DataFrame.
    """

    fingerprint_distance = FingerprintDistance()
    feature_weights_calculated = fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

    assert feature_weights_calculated.dtypes.weight == weight_column_dtype
    assert feature_weights_calculated.dtypes.feature_name == feature_name_column_dtype
    assert feature_weights_calculated.shape == shape

    weights_calculated = feature_weights_calculated.weight
    weights = pd.Series(list(feature_weights.values()))

    assert weights_calculated.equals(weights)


@pytest.mark.parametrize('feature_weights', [
    ({'size': 0.1}),  # Features missing
    ({'size': 0.1, 'xxx': 0.1}),  # Unknown feature
    ({
        'size': 0.5,
        'hbd': 0.0625,
        'hba': 0.0625,
        'charge': 0.0625,
        'aromatic': 0.0625,
        'aliphatic': 0.0625,
        'sco': 0.0625,
        'exposure': 0.0625,
        'distance_to_centroid': 0.125,
        'distance_to_hinge_region': 0.125,
        'distance_to_dfg_region': 0.125,
        'distance_to_front_pocket': 0.125,
        'moment1': 0.0,
        'moment2': 0.0,
        'moment3': 0.0
    }),  # Weights do not sum up to 1.0
])
def test_format_weight_per_feature_valueerror(feature_weights):
    """
    Test if incorrect input feature weights raise ValueError.

    Parameters
    ----------
    feature_weights : dict of floats
        Dictionary does not fulfill one or more of these conditions:
        Weights per feature which need to sum up to 1.0.
        Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure, distance_to_centroid,
        distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
    """

    with pytest.raises(ValueError):
        fingerprint_distance = FingerprintDistance()
        fingerprint_distance._format_weight_per_feature(feature_weights)


@pytest.mark.parametrize('feature_weights', [
    ({
        'size': 'bla',
        'hbd': 0.0625,
        'hba': 0.0625,
        'charge': 0.0625,
        'aromatic': 0.0625,
        'aliphatic': 0.0625,
        'sco': 0.0625,
        'exposure': 0.0625,
        'distance_to_centroid': 0.125,
        'distance_to_hinge_region': 0.125,
        'distance_to_dfg_region': 0.125,
        'distance_to_front_pocket': 0.125,
        'moment1': 0.0,
        'moment2': 0.0,
        'moment3': 0.0
    }),  # Weight value is not float
    ({
        'size': 1,
        'hbd': 0.0,
        'hba': 0.0,
        'charge': 0.0,
        'aromatic': 0.0,
        'aliphatic': 0.0,
        'sco': 0.0,
        'exposure': 0.0,
        'distance_to_centroid': 0.0,
        'distance_to_hinge_region': 0.0,
        'distance_to_dfg_region': 0.0,
        'distance_to_front_pocket': 0.0,
        'moment1': 0.0,
        'moment2': 0.0,
        'moment3': 0.0
    }),  # Weight value is not float
])
def test_format_weight_per_feature_typeerror(feature_weights):
    """
    Test if incorrect input feature weights raise TypeError.

    Parameters
    ----------
    feature_weights : dict of float
        Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure, distance_to_centroid,
        distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
        Set dictionary values to data type other than float.
    """

    with pytest.raises(TypeError):
        fingerprint_distance = FingerprintDistance()
        fingerprint_distance._format_weight_per_feature(feature_weights)


@pytest.mark.parametrize('feature_weights, weight_column_dtype, feature_name_column_dtype, shape', [
    (
        {
            'size': 1.0,
            'hbd': 0.0,
            'hba': 0.0,
            'charge': 0.0,
            'aromatic': 0.0,
            'aliphatic': 0.0,
            'sco': 0.0,
            'exposure': 0.0,
            'distance_to_centroid': 0.0,
            'distance_to_hinge_region': 0.0,
            'distance_to_dfg_region': 0.0,
            'distance_to_front_pocket': 0.0,
            'moment1': 0.0,
            'moment2': 0.0,
            'moment3': 0.0
        },
        'float64',
        'object',
        (15, 2)
    )
])
def test_format_weight_per_feature(feature_weights, weight_column_dtype, feature_name_column_dtype, shape):
    """
    Test formatting of weights per feature type (weights need to be transformed into a DataFrame).

    Parameters
    ----------
    feature_weights : dict of float or None (15 items)
        Weights per feature which need to sum up to 1.0.
        Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure, distance_to_centroid,
        distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
        Default feature weights (None) are set equally distributed to 1/15 (15 feature in total).
    weight_column_dtype : str
        Data type of weight returned DataFrame column.
    feature_name_column_dtype : str
        Data type of feature_name returned DataFrame column.
    shape : tuple
        Dimension of returned DataFrame.
    """

    fingerprint_distance = FingerprintDistance()
    feature_weights_calculated = fingerprint_distance._format_weight_per_feature(feature_weights)

    assert feature_weights_calculated.dtypes.weight == weight_column_dtype
    assert feature_weights_calculated.dtypes.feature_name == feature_name_column_dtype
    assert feature_weights_calculated.shape == shape

    weights_calculated = feature_weights_calculated.weight
    weights = pd.Series(list(feature_weights.values()))

    assert weights_calculated.equals(weights)
