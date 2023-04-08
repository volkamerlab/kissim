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
        assert isinstance(feature_distances_generator.data, pd.DataFrame)
        assert feature_distances_generator.data.columns.to_list() == [
            "structure.1",
            "structure.2",
            "kinase.1",
            "kinase.2",
        ] + [f"distance.{i}" for i in range(1, 16)] + [f"bit_coverage.{i}" for i in range(1, 16)]
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
        assert isinstance(feature_distances_generator.data, pd.DataFrame)
        assert isinstance(feature_distances_generator.structure_kinase_ids, list)

    @pytest.mark.parametrize(
        "structure_kinase_ids",
        [[["pdbA", "kinaseA"], ["pdbB", "kinaseA"], ["pdbC", "kinaseB"]]],
    )
    def test_structure_kinase_ids(self, feature_distances_generator, structure_kinase_ids):
        assert feature_distances_generator._structure_kinase_ids == structure_kinase_ids

    @pytest.mark.parametrize(
        "structure_pair_ids", [[["pdbA", "pdbB"], ["pdbA", "pdbC"], ["pdbB", "pdbC"]]]
    )
    def test_structure_pair_ids(self, feature_distances_generator, structure_pair_ids):
        assert feature_distances_generator.structure_pair_ids == structure_pair_ids

    @pytest.mark.parametrize(
        "kinase_pair_ids",
        [[["kinaseA", "kinaseA"], ["kinaseA", "kinaseB"], ["kinaseA", "kinaseB"]]],
    )
    def test_kinase_pair_ids(self, feature_distances_generator, kinase_pair_ids):
        assert feature_distances_generator.kinase_pair_ids == kinase_pair_ids

    @pytest.mark.parametrize("structure_ids", [["pdbA", "pdbB", "pdbC"]])
    def test_structure_ids(self, feature_distances_generator, structure_ids):
        assert feature_distances_generator.structure_ids == structure_ids

    @pytest.mark.parametrize("kinase_ids", [["kinaseA", "kinaseB"]])
    def test_kinase_ids(self, feature_distances_generator, kinase_ids):
        assert feature_distances_generator.kinase_ids == kinase_ids

    def test_to_from_csv(self, feature_distances_generator):
        with enter_temp_directory():
            filepath = Path("test.csv.bz2")

            feature_distances_generator.to_csv(filepath)
            assert filepath.exists()

            feature_distances_generator_from_csv = FeatureDistancesGenerator.from_csv(filepath)
            assert isinstance(feature_distances_generator_from_csv, FeatureDistancesGenerator)

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
