"""
Unit and regression test for the kissim.comparison.FingerprintDistanceGenerator class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.comparison import FingerprintDistance, FingerprintDistanceGenerator


PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestsFingerprintDistanceGenerator:
    """
    Test FingerprintDistanceGenerator class methods.
    """

    @pytest.mark.parametrize(
        "feature_weights, structure_ids, kinase_ids, structure_kinase_ids",
        [
            (
                None,
                "pdb1 pdb2 pdb3".split(),
                "kinase1 kinase2".split(),
                [("pdb1", "kinase1"), ("pdb2", "kinase1"), ("pdb3", "kinase2")],
            )
        ],
    )
    def test_from_feature_distances_generator(
        self,
        feature_distances_generator,
        feature_weights,
        structure_ids,
        kinase_ids,
        structure_kinase_ids,
    ):
        """
        Test FingerprintDistanceGenerator class attributes.
        """

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = (
            FingerprintDistanceGenerator.from_feature_distances_generator(
                feature_distances_generator
            )
        )

        # Test attributes
        assert fingerprint_distance_generator.structure_kinase_ids == structure_kinase_ids
        assert isinstance(fingerprint_distance_generator.feature_weights, np.ndarray)
        assert len(fingerprint_distance_generator.feature_weights) == 15
        assert isinstance(fingerprint_distance_generator._structures1, list)
        assert isinstance(fingerprint_distance_generator._structures2, list)
        assert isinstance(fingerprint_distance_generator._kinases1, list)
        assert isinstance(fingerprint_distance_generator._kinases2, list)
        assert isinstance(fingerprint_distance_generator._distances, np.ndarray)
        assert isinstance(fingerprint_distance_generator._bit_coverages, np.ndarray)

        # Test properties
        assert fingerprint_distance_generator.structure_ids == structure_ids
        assert fingerprint_distance_generator.kinase_ids == kinase_ids
        assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)
        data_columns = "structure1 structure2 kinase1 kinase2 distance coverage".split()
        assert list(fingerprint_distance_generator.data.columns) == data_columns

    @pytest.mark.parametrize(
        "structure_klifs_ids, klifs_session, n_cores, feature_weights",
        [
            ([110, 118], REMOTE, 1, None),
            ([110, 118], REMOTE, 2, None),
            ([110, 118], LOCAL, 1, None),
            ([110, 118], LOCAL, 2, None),
            ([110, 118], None, None, None),
        ],
    )
    def test_from_structure_klifs_ids(
        self, structure_klifs_ids, klifs_session, n_cores, feature_weights
    ):
        """
        Test FeatureDistancesGenerator class attributes.
        """

        # Test FeatureDistancesGenerator class attributes
        feature_distances_generator = FingerprintDistanceGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores, feature_weights
        )
        assert isinstance(feature_distances_generator, FingerprintDistanceGenerator)

    def test_get_fingerprint_distance(self, feature_distances):
        """
        Test if return type is FingerprintDistance class instance.

        Parameters
        ----------
        feature_distances : kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_calculated = fingerprint_distance_generator._get_fingerprint_distance(
            feature_distances
        )

        assert isinstance(fingerprint_distance_calculated, FingerprintDistance)

    def test_get_fingerprint_distance_from_list(self, feature_distances_generator):
        """
        Test if return type is instance of list of FingerprintDistance class instances.

        Parameters
        ----------
        feature_distances_generator : FeatureDistancesGenerator
            Feature distances for multiple fingerprints.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator()
        fingerprint_distance_list = (
            fingerprint_distance_generator._get_fingerprint_distance_from_list(
                fingerprint_distance_generator._get_fingerprint_distance,
                list(feature_distances_generator.data.values()),
            )
        )

        assert isinstance(fingerprint_distance_list, list)

        for i in fingerprint_distance_list:
            assert isinstance(i, FingerprintDistance)

    @pytest.mark.parametrize(
        "structure_distance_matrix",
        [
            pd.DataFrame(
                [[0.0, 0.75, 1.0], [0.75, 0.0, 0.8], [1.0, 0.8, 0.0]],
                columns="pdb1 pdb2 pdb3".split(),
                index="pdb1 pdb2 pdb3".split(),
            )
        ],
    )
    def test_structure_distance_matrix(
        self, fingerprint_distance_generator, structure_distance_matrix
    ):
        """
        Test if structure distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        structure_distance_matrix : pandas.DataFrame
            Structure distance matrix.
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = (
            fingerprint_distance_generator.structure_distance_matrix()
        )

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)

    @pytest.mark.parametrize(
        "by, kinase_distance_matrix",
        [
            (
                "minimum",
                pd.DataFrame(
                    [[0.0, 0.8], [0.8, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Minimum
            (
                "maximum",
                pd.DataFrame(
                    [[0.75, 1.0], [1.0, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Maximum
            (
                "mean",
                pd.DataFrame(
                    [[0.25, 0.9], [0.9, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Mean
        ],
    )
    def test_kinase_distance_matrix(
        self, fingerprint_distance_generator, by, kinase_distance_matrix
    ):
        """
        Test if kinase distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.
        fill : bool
            Fill or fill not (default) lower triangle of distance matrix.
        kinase_distance_matrix : pandas.DataFrame
            xxx
        """

        # Test generation of structure distance matrix
        kinase_distance_matrix_calculated = fingerprint_distance_generator.kinase_distance_matrix(
            by
        )

        assert kinase_distance_matrix_calculated.equals(kinase_distance_matrix)
