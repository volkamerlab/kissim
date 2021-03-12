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

        # Test class attributes
        assert fingerprint_distance.structure_pair_ids == feature_distances.structure_pair_ids
        assert fingerprint_distance.kinase_pair_ids == feature_distances.kinase_pair_ids
        assert np.isclose(fingerprint_distance.distance, distance, rtol=1e-04)
        assert np.isclose(fingerprint_distance.bit_coverage, coverage, rtol=1e-04)
        assert np.array_equal(fingerprint_distance.feature_weights, np.array(feature_weights))
