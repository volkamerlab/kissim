"""
Unit and regression test for kinsim_structure.similarity functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinsim_structure.similarity import FingerprintDistance


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

    fingerprint_distance = FingerprintDistance()
    feature_weights_calculated = fingerprint_distance._format_weight_per_feature(feature_weights)

    assert feature_weights_calculated.dtypes.weight == weight_column_dtype
    assert feature_weights_calculated.dtypes.feature_name == feature_name_column_dtype
    assert feature_weights_calculated.shape == shape

    weights_calculated = feature_weights_calculated.weight
    weights = pd.Series(list(feature_weights.values()))

    assert weights_calculated.equals(weights)
