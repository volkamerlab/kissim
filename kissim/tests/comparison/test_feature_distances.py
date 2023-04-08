"""
Unit and regression test for the kissim.comparison.FeatureDistances class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kissim.comparison import FeatureDistances

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsFeatureDistances:
    """
    Test FeatureDistances class methods.
    """

    @pytest.mark.parametrize(
        "feature_pair, distance_measure, distance",
        [
            (np.array([[4, 0], [0, 3]]), "scaled_euclidean", 2.5),
            (np.array([]), "scaled_euclidean", np.nan),
        ],
    )
    def test_calculate_feature_distance(self, feature_pair, distance_measure, distance):
        """
        Test distance calculation for two value (feature) lists.

        Parameters
        ----------
        feature_pair : np.ndarray
            Pairwise bits of one feature extracted from two fingerprints (only bit positions
            without any NaN value).
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        distance : float
            Distance between two value lists.
        """

        feature_distances = FeatureDistances()
        distance_calculated = feature_distances._calculate_feature_distance(
            feature_pair, distance_measure
        )

        if np.isnan(distance):
            assert np.isnan(distance_calculated)
        else:
            assert np.isclose(distance_calculated, distance, rtol=1e-04)

    @pytest.mark.parametrize(
        "feature_pair, distance_measure",
        [("feature_pair", "scaled_euclidean")],  # Feature pair is not np.ndarray
    )
    def test_calculate_feature_distance_typeerror(self, feature_pair, distance_measure):
        """
        Test TypeError exceptions in distance calculation for two value (feature) lists.

        Parameters
        ----------
        feature_pair : np.ndarray
            Pairwise bits of one feature extracted from two fingerprints (only bit positions
            without any NaN value).
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        """

        with pytest.raises(TypeError):
            feature_distance = FeatureDistances()
            feature_distance._calculate_feature_distance(feature_pair, distance_measure)

    @pytest.mark.parametrize(
        "feature_pair, distance_measure",
        [
            (np.array([[1, 2], [1, 2]]), "xxx"),  # Distance measure is not implemented
            (
                np.array([[1, 2], [1, 2], [1, 2]]),
                "scaled_euclidean",
            ),  # Feature pair has more than two rows
            (np.array([[1, 2], [1, 2]]), 11),  # Distance measure is not str
        ],
    )
    def test_calculate_feature_distance_valueerror(self, feature_pair, distance_measure):
        """
        Test ValueError exceptions in distance calculation for two value (feature) lists.

        Parameters
        ----------
        feature_pair : np.ndarray
            Pairwise bits of one feature extracted from two fingerprints (only bit positions
            without any NaN value).
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        """

        with pytest.raises(ValueError):
            feature_distance = FeatureDistances()
            feature_distance._calculate_feature_distance(feature_pair, distance_measure)

    @pytest.mark.parametrize(
        "feature1, feature2, distance, bit_coverage",
        [
            (pd.Series([1, 1, 1, 1]), pd.Series([0, 0, 0, 0]), 0.5, 1.0),
            (pd.Series([1, 1, 1, 1, np.nan]), pd.Series([0, 0, 0, 0, 0]), 0.5, 0.8),
            (pd.Series([1, 1, 1, 1, 1]), pd.Series([0, 0, 0, 0, np.nan]), 0.5, 0.8),
            (pd.Series([1, 1, 1, 1, np.nan]), pd.Series([0, 0, 0, 0, np.nan]), 0.5, 0.8),
        ],
    )
    def test_get_feature_distances_and_bit_coverages(
        self, feature1, feature2, distance, bit_coverage
    ):
        """
        Test if feature distance and bit coverage is correct for given feature bits.

        Parameters
        ----------
        feature1 : pd.Series
            Feature bits for a given feature in fingerprint 1.
        feature2 : pd.Series
            Feature bits for a given feature in fingerprint 2.
        distance : float
            Distance value for a feature pair.
        bit_coverage : float
            Bit coverage value for a feature pair.
        """

        feature_distances = FeatureDistances()
        (
            distance_calculated,
            bit_coverage_calculated,
        ) = feature_distances._get_feature_distances_and_bit_coverages(feature1, feature2)

        assert np.isclose(distance_calculated, distance, rtol=1e-04)
        assert np.isclose(bit_coverage_calculated, bit_coverage, rtol=1e-04)

    @pytest.mark.parametrize(
        "feature1, feature2", [(pd.Series([1, 1, 1, 1]), pd.Series([0, 0, 0]))]
    )
    def test_get_feature_distances_and_bit_coverages_valueerror(self, feature1, feature2):
        """
        Test ValueError exceptions in feature distance calculation.

        Parameters
        ----------
        feature1 : np.ndarray
            Feature bits for a given feature in fingerprint 1.
        feature2 : np.ndarray
            Feature bits for a given feature in fingerprint 2.
        """

        feature_distances = FeatureDistances()

        with pytest.raises(ValueError):
            feature_distances._get_feature_distances_and_bit_coverages(feature1, feature2)

    def test_from_fingerprints(self, fingerprint_generator):
        """
        Test data type and dimensions of feature distances between two fingerprints.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Fingerprints
        fingerprints = list(fingerprint_generator.data.values())

        # Get feature distances
        feature_distances = FeatureDistances.from_fingerprints(
            fingerprint1=fingerprints[0], fingerprint2=fingerprints[1]
        )

        # Class attribute types and dimensions correct?
        assert isinstance(feature_distances.structure_pair_ids, tuple)
        assert len(feature_distances.structure_pair_ids) == 2

        assert isinstance(feature_distances.distances, np.ndarray)
        assert len(feature_distances.distances) == 15

        assert isinstance(feature_distances.bit_coverages, np.ndarray)
        assert len(feature_distances.bit_coverages) == 15

        # Class property type and dimension correct?
        assert isinstance(feature_distances.data, pd.DataFrame)

        feature_type_dimension_calculated = feature_distances.data.groupby(
            by="feature_type", sort=False
        ).size()
        feature_type_dimension = pd.Series(
            [8, 4, 3], index="physicochemical distances moments".split()
        )
        assert all(feature_type_dimension_calculated == feature_type_dimension)

    @pytest.mark.parametrize(
        "feature_distances_dict",
        [
            {
                "structure_pair_ids": ["pdbA", "pdbB"],
                "kinase_pair_ids": ["kinaseA", "kinaseB"],
                "distances": [1.0] * 15,
                "bit_coverages": [1.0] * 15,
            }
        ],
    )
    def test_from_dict(self, feature_distances_dict):
        feature_distances_calculated = FeatureDistances._from_dict(feature_distances_dict)
        assert isinstance(feature_distances_calculated, FeatureDistances)
        assert isinstance(feature_distances_calculated.structure_pair_ids, tuple)
        assert isinstance(feature_distances_calculated.kinase_pair_ids, tuple)
        assert isinstance(feature_distances_calculated.distances, np.ndarray)
        assert isinstance(feature_distances_calculated.bit_coverages, np.ndarray)
