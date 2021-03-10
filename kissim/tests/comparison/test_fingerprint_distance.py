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
        "feature_weights, feature_weights_formatted",
        [
            (None, np.array([0.0667] * 15)),
            (
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                np.array(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
            ),
            ([1.0, 0.0, 0.0], np.array([0.125] * 8 + [0.0] * 7)),
        ],
    )
    def test_format_weights(self, feature_weights, feature_weights_formatted):
        """
        Test if feature weights are added correctly to feature distance DataFrame.

        Parameters
        ----------
        feature_weights : None or list of float
            Feature weights.
        feature_weights_formatted : list of float
            Formatted feature weights of length 15.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        feature_weights_formatted_calculated = fingerprint_distance._format_weights(
            feature_weights
        )

        assert np.isclose(
            np.std(feature_weights_formatted),
            np.std(feature_weights_formatted_calculated),
            rtol=1e-04,
        )

    @pytest.mark.parametrize("feature_weights", [{"a": 0}, "bla"])
    def test_format_weights_typeerror(self, feature_weights):
        """
        Test if wrong data type of input feature weights raises TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weights(feature_weights)

    @pytest.mark.parametrize(
        "feature_weights",
        [
            [0],
        ],
    )
    def test_format_weights_valueerror(self, feature_weights):
        """
        Test if wrong data type of input feature weights raises TypeError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weights(feature_weights)

    @pytest.mark.parametrize(
        "feature_type_weights, feature_weights",
        [
            (
                [0.0, 1.0, 0.0],
                np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]
                ),
            )
        ],
    )
    def test_format_weight_per_feature_type(self, feature_type_weights, feature_weights):
        """
        Test formatting of weights per feature type (weights need to be equally distributed
        between all features in feature type and transformed into a DataFrame).

        Parameters
        ----------
        feature_type_weights : dict of float (3 items) or None
            Weights per feature type which need to sum up to 1.0.
        feature_weights : dict of float (15 items) or None
            Weights per feature which need to sum up to 1.0.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        feature_weights_calculated = fingerprint_distance._format_weight_per_feature_type(
            feature_type_weights
        )

        # Test weight values
        assert np.isclose(np.std(feature_weights_calculated), np.std(feature_weights), rtol=1e-04)

    @pytest.mark.parametrize(
        "feature_type_weights",
        [
            ([0.1]),  # Features missing
            ([0.5, 0.5, 0.5]),  # Weights do not sum up to 1.0
        ],
    )
    def test_format_weight_per_feature_type_valueerror(self, feature_type_weights):
        """
        Test if incorrect input feature type weights raise ValueError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

    @pytest.mark.parametrize(
        "feature_type_weights",
        [
            ({"a": 1.0}),  # Input is no list
        ],
    )
    def test_format_weight_per_feature_type_typeerror(self, feature_type_weights):
        """
        Test if incorrect input feature type weights raise TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature_type(feature_type_weights)

    @pytest.mark.parametrize(
        "feature_weights, feature_weights_formatted",
        [
            (
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                np.array(
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
            ),
            (None, [0.0667] * 15),
        ],
    )
    def test_format_weight_per_feature(self, feature_weights, feature_weights_formatted):
        """
        Test formatting of weights per feature type (weights need to be transformed into a
        DataFrame).

        Parameters
        ----------
        feature_weights : dict of float or None (15 items)
            Weights per feature which need to sum up to 1.0.
        feature_weights_formatted : xxx
            Formatted feature weights.
        """

        # FingerprintDistance
        fingerprint_distance = FingerprintDistance()
        feature_weights_formatted_calculated = fingerprint_distance._format_weight_per_feature(
            feature_weights
        )

        assert np.isclose(
            np.std(feature_weights_formatted_calculated),
            np.std(feature_weights_formatted),
            rtol=1e-04,
        )

    @pytest.mark.parametrize(
        "feature_weights",
        [
            ([0.1]),  # Features missing
            (
                [
                    0.5,
                    0.0625,
                    0.0625,
                    0.0625,
                    0.0625,
                    0.0625,
                    0.0625,
                    0.0625,
                    0.125,
                    0.125,
                    0.125,
                    0.125,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),  # Weights do not sum up to 1.0
        ],
    )
    def test_format_weight_per_feature_valueerror(self, feature_weights):
        """
        Test if incorrect input feature weights raise ValueError.
        """

        with pytest.raises(ValueError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature(feature_weights)

    @pytest.mark.parametrize("feature_weights", [("is_string")])  # Input is no list
    def test_format_weight_per_feature_typeerror(self, feature_weights):
        """
        Test if incorrect input feature weights raise TypeError.
        """

        with pytest.raises(TypeError):
            fingerprint_distance = FingerprintDistance()
            fingerprint_distance._format_weight_per_feature(feature_weights)

    @pytest.mark.parametrize(
        "feature_weights, distance, coverage",
        [
            (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.25],
                0.5,
                0.75,
            )
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

        # Test class attributes:

        # Molecule codes
        assert fingerprint_distance.structure_pair_ids == feature_distances.structure_pair_ids

        # Fingerprint distance
        assert np.isclose(fingerprint_distance.distance, distance, rtol=1e-04)

        # Fingerprint coverage
        assert np.isclose(fingerprint_distance.bit_coverage, coverage, rtol=1e-04)
