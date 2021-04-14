"""
Unit and regression test for the kissim.comparison.FeatureDistancesGenerator class.
"""
from pathlib import Path

import pytest
import pandas as pd
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.utils import enter_temp_directory
from kissim.encoding import Fingerprint
from kissim.comparison import FeatureDistances, FeatureDistancesGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestsFeatureDistancesGenerator:
    """
    Test FeatureDistancesGenerator class methods.
    """

    @pytest.mark.parametrize(
        "feature_weights, structure_ids, kinase_ids",
        [
            (
                None,
                ["HUMAN/ABL1_2g2i_chainA", "HUMAN/AAK1_4wsq_altA_chainB"],
                ["AAK1", "ABL1"],
            )
        ],
    )
    def test_from_fingerprints(
        self,
        fingerprint_generator,
        feature_weights,
        structure_ids,
        kinase_ids,
    ):
        """
        Test FeatureDistancesGenerator class attributes.
        """

        # Test FeatureDistancesGenerator class attributes
        feature_distances_generator = FeatureDistancesGenerator.from_fingerprint_generator(
            fingerprint_generator
        )
        assert isinstance(feature_distances_generator, FeatureDistancesGenerator)

        # Test attributes
        assert isinstance(feature_distances_generator.data, list)
        assert isinstance(feature_distances_generator.data[0], FeatureDistances)
        assert isinstance(feature_distances_generator.structure_kinase_ids, list)

    @pytest.mark.parametrize(
        "structure_klifs_ids, klifs_session, n_cores",
        [
            ([110, 118], REMOTE, 1),
            ([110, 118], REMOTE, 2),
            ([110, 118], LOCAL, 1),
            ([110, 118], LOCAL, 2),
            ([110, 118], None, None),
        ],
    )
    def test_from_structure_klifs_ids(self, structure_klifs_ids, klifs_session, n_cores):

        # Test FeatureDistancesGenerator class attributes
        feature_distances_generator = FeatureDistancesGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores
        )
        assert isinstance(feature_distances_generator, FeatureDistancesGenerator)

        # Test attributes
        assert isinstance(feature_distances_generator.data, list)
        assert isinstance(feature_distances_generator.data[0], FeatureDistances)
        assert isinstance(feature_distances_generator.structure_kinase_ids, list)

    def test_to_from_json(self, feature_distances_generator):

        with enter_temp_directory():

            json_filepath = Path("test.json")

            feature_distances_generator.to_json(json_filepath)
            assert json_filepath.exists()

            feature_distances_generator_from_json = FeatureDistancesGenerator.from_json(
                json_filepath
            )
            assert isinstance(feature_distances_generator_from_json, FeatureDistancesGenerator)

    @pytest.mark.parametrize("structure_ids", [["pdb1", "pdb2", "pdb3"]])
    def test_structure_ids(self, feature_distances_generator, structure_ids):

        structure_ids_calculated = feature_distances_generator.structure_ids
        assert structure_ids_calculated == structure_ids

    @pytest.mark.parametrize("kinase_ids", [["kinase1", "kinase2"]])
    def test_kinase_ids(self, feature_distances_generator, kinase_ids):

        kinase_ids_calculated = feature_distances_generator.kinase_ids
        assert kinase_ids_calculated == kinase_ids

    @pytest.mark.parametrize(
        "fingerprints, pairs",
        [
            (
                {"a": Fingerprint(), "b": Fingerprint(), "c": Fingerprint()},
                [("a", "b"), ("a", "c"), ("b", "c")],
            )
        ],
    )
    def test_fingerprint_pairs(self, fingerprints, pairs):
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

    @pytest.mark.parametrize("structure_id1, structure_id2", [("pdb1", "pdb3")])
    def test_by_structure_pair(self, feature_distances_generator, structure_id1, structure_id2):

        feature_distances_data = feature_distances_generator.by_structure_pair(
            structure_id1, structure_id2
        )
        assert isinstance(feature_distances_data, pd.DataFrame)
        assert feature_distances_data.columns.to_list() == [
            "feature_type",
            "feature_name",
            "distance",
            "bit_coverage",
        ]

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
            generator._get_feature_distances, fingerprint_generator.data, 1
        )

        assert isinstance(feature_distances_list, list)

        for i in feature_distances_list:
            assert isinstance(i, FeatureDistances)
