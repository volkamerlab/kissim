"""
Unit and regression test for the kissim.comparison.FeatureDistancesGenerator class.
"""
from pathlib import Path

import pytest

from kissim.encoding import Fingerprint
from kissim.comparison import FeatureDistances, FeatureDistancesGenerator
from kissim.tests.comparison.fixures import fingerprint_generator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsFeatureDistancesGenerator:
    """
    Test FeatureDistancesGenerator class methods.
    """

    @pytest.mark.parametrize(
        "fingerprints, pairs",
        [
            (
                {"a": Fingerprint(), "b": Fingerprint(), "c": Fingerprint()},
                [("a", "b"), ("a", "c"), ("b", "c")],
            )
        ],
    )
    def test_get_fingerprint_pairs(self, fingerprints, pairs):
        """
        Test calculation of all fingerprint pair combinations from fingerprints dictionary.

        Parameters
        ----------
        fingerprints : dict of kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        pairs : list of list of str
            List of molecule code pairs (list).
        """

        generator = FeatureDistancesGenerator()
        pairs_calculated = generator._fingerprint_pairs(fingerprints)

        for pair_calculated, pair in zip(pairs_calculated, pairs):
            assert pair_calculated == pair

    def test_get_feature_distances(self, fingerprint_generator):
        """
        Test if return type is instance of FeatureDistance class.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Get fingerprint pair from FingerprintGenerator
        pair = list(fingerprint_generator.data.keys())[:2]
        fingerprints = fingerprint_generator.data

        # Test feature distance calculation
        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_calculated = feature_distances_generator._get_feature_distances(
            pair, fingerprints
        )

        assert isinstance(feature_distances_calculated, FeatureDistances)

    def test_get_feature_distances_from_list(self, fingerprint_generator):
        """
        Test if return type is instance of list of FeatureDistance class.

        Parameters
        ----------
        fingerprint_generator : FingerprintGenerator
            Multiple fingerprints.
        """

        # Test bulk feature distance calculation
        generator = FeatureDistancesGenerator()

        feature_distances_list = generator._get_feature_distances_from_list(
            generator._get_feature_distances, fingerprint_generator.data
        )

        assert isinstance(feature_distances_list, list)

        for i in feature_distances_list:
            assert isinstance(i, FeatureDistances)

    @pytest.mark.parametrize(
        "distance_measure, feature_weights, structure_ids, kinase_ids",
        [
            (
                "scaled_euclidean",
                None,
                ["HUMAN/ABL1_2g2i_chainA", "HUMAN/AAK1_4wsq_altA_chainB"],
                ["AAK1", "ABL1"],
            )
        ],
    )
    def test_from_fingerprints(
        self,
        fingerprint_generator,
        distance_measure,
        feature_weights,
        structure_ids,
        kinase_ids,
    ):
        """
        Test FeatureDistancesGenerator class attributes.

        Parameters
        ----------
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        """

        # Test FeatureDistancesGenerator class attributes
        feature_distances_generator = FeatureDistancesGenerator.from_fingerprint_generator(
            fingerprint_generator
        )

        # Test attributes
        assert isinstance(feature_distances_generator.data, dict)

        # Test example value from dictionary
        example_key = list(feature_distances_generator.data.keys())[0]
        assert isinstance(feature_distances_generator.data[example_key], FeatureDistances)
