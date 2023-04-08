"""
Unit and regression test for the kissim.comparison.FingerprintDistance class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kissim.comparison import FingerprintDistance

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsFingerprintDistance:
    """
    Test FingerprintDistance class methods.
    """

    @pytest.mark.parametrize(
        "feature_weights, distance, coverage",
        [
            (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
                0.5,
                0.75,
            ),
        ],
    )
    def test_from_feature_distances(self, feature_distances, feature_weights, distance, coverage):
        """
        Test if fingerprint distances are calculated correctly based on feature distances.

        Parameters
        ----------
        feature_distances : kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        feature_weights : dict of float or None
            Feature weights.
        distance : float
            Fingerprint distance.
        coverage : float
            Fingerprint coverage.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance.from_feature_distances(
            feature_distances, feature_weights
        )

        # Test class attributes
        assert fingerprint_distance.structure_pair_ids == feature_distances.structure_pair_ids
        assert fingerprint_distance.kinase_pair_ids == feature_distances.kinase_pair_ids
        assert np.isclose(fingerprint_distance.distance, distance, rtol=1e-04)
        assert np.isclose(fingerprint_distance.bit_coverage, coverage, rtol=1e-04)

    @pytest.mark.parametrize(
        "values, weights, calculated_weighted_sum",
        [
            (
                np.array([0.1, 0.2]),
                np.array([0.5, 0.5]),
                0.15,
            ),
            (
                np.array([0.1, 0.2]),
                np.array([1.0, 0.0]),
                0.1,
            ),
            (
                np.array([0.1, 0.2]),
                np.array([0.2, 0.8]),
                0.18,
            ),
        ],
    )
    def test_calculate_weighted_sum(self, values, weights, calculated_weighted_sum):
        fingerprint_distance = FingerprintDistance()
        calculated_weighted_sum_calculated = fingerprint_distance._calculate_weighted_sum(
            values, weights
        )
        assert np.isclose(calculated_weighted_sum_calculated, calculated_weighted_sum, rtol=1e-04)

    @pytest.mark.parametrize(
        "values, weights",
        [
            (
                np.array([0.1, np.nan]),  # Values contain NaN
                np.array([0.5, 0.5]),
            ),
            (
                np.array([0.1, 0.1]),
                np.array([0.5, 0.0]),  # Sum is not 1.0
            ),
        ],
    )
    def test_calculate_weighted_sum_raises(self, values, weights):
        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._calculate_weighted_sum(values, weights)

    @pytest.mark.parametrize(
        "distances, weights, distances_wo_nan, weights_wo_nan_recalibrated",
        [
            (
                np.array([np.nan, 0.1, 0.2, 0.8, 0.9]),
                np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                np.array([0.1, 0.2, 0.8, 0.9]),
                np.array([0.25, 0.25, 0.25, 0.25]),
            ),
            (
                np.array([np.nan, 0.1, 0.2, 0.8, 0.9]),
                np.array([0.1, 0.2, 0.3, 0.3, 0.1]),
                np.array([0.1, 0.2, 0.8, 0.9]),
                np.array([0.2, 0.3, 0.3, 0.1]) + np.array([0.2, 0.3, 0.3, 0.1]) * 0.1 / 0.9,
            ),
        ],
    )
    def test_remove_nan_distances_and_recalibrate_weights(
        self, distances, weights, distances_wo_nan, weights_wo_nan_recalibrated
    ):
        fingerprint_distance = FingerprintDistance()
        (
            distances_wo_nan_calculated,
            weights_wo_nan_recalibrated_calculated,
        ) = fingerprint_distance._remove_nan_distances_and_recalibrate_weights(distances, weights)
        assert np.array_equal(distances_wo_nan_calculated, distances_wo_nan)
        assert np.array_equal(weights_wo_nan_recalibrated_calculated, weights_wo_nan_recalibrated)
