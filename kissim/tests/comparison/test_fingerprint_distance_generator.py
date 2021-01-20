"""
Unit and regression test for the kissim.comparison.FingerprintDistanceGenerator class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kissim.comparison import FingerprintDistance, FingerprintDistanceGenerator
from kissim.tests.comparison.fixures import (
    feature_distances,
    feature_distances_generator,
    fingerprint_distance_generator,
)

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


class TestsFingerprintDistanceGenerator:
    """
    Test FingerprintDistanceGenerator class methods.
    """

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
        "distance_measure, feature_weights, molecule_codes, kinase_names",
        [
            (
                "scaled_euclidean",
                None,
                "HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split(),
                "kinase1 kinase2".split(),
            )
        ],
    )
    def test_from_feature_distances_generator(
        self,
        feature_distances_generator,
        distance_measure,
        feature_weights,
        molecule_codes,
        kinase_names,
    ):
        """
        Test FingerprintDistanceGenerator class attributes.

        Parameters
        ----------
        feature_distances_generator : FeatureDistancesGenerator
            Feature distances for multiple fingerprints.
        distance_measure : str
            Type of distance measure, defaults to Euclidean distance.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 feature in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region,
                distance_to_front_pocket, moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.
        molecule_codes : list of str
            List of molecule codes associated with input fingerprints.
        kinase_names : list of str
            List of kinase names associated with input fingerprints.
        """

        # FingerprintDistanceGenerator
        fingerprint_distance_generator = FingerprintDistanceGenerator()
        print(feature_distances_generator.data)
        fingerprint_distance_generator.from_feature_distances_generator(
            feature_distances_generator
        )

        # Test attributes
        assert fingerprint_distance_generator.distance_measure == distance_measure
        assert fingerprint_distance_generator.feature_weights == feature_weights
        assert fingerprint_distance_generator.molecule_codes == molecule_codes
        assert fingerprint_distance_generator.kinase_names == kinase_names

        assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)

        data_columns = "molecule_code_1 molecule_code_2 distance coverage".split()
        assert list(fingerprint_distance_generator.data.columns) == data_columns

    @pytest.mark.parametrize(
        "fill, structure_distance_matrix",
        [
            (
                False,
                pd.DataFrame(
                    [[0.0, 0.5, 0.75], [np.nan, 0.0, 1.0], [np.nan, np.nan, 0.0]],
                    columns="HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split(),
                    index="HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split(),
                ),
            ),
            (
                True,
                pd.DataFrame(
                    [[0.0, 0.5, 0.75], [0.5, 0.0, 1.0], [0.75, 1.0, 0.0]],
                    columns="HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split(),
                    index="HUMAN/kinase1_pdb1 HUMAN/kinase1_pdb2 HUMAN/kinase2_pdb1".split(),
                ),
            ),
        ],
    )
    def test_get_structure_distance_matrix(
        self, fingerprint_distance_generator, fill, structure_distance_matrix
    ):
        """
        Test if structure distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        fill
        structure_distance_matrix
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = (
            fingerprint_distance_generator.get_structure_distance_matrix(fill)
        )

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)

    @pytest.mark.parametrize(
        "by, fill, structure_distance_matrix",
        [
            (
                "minimum",
                False,
                pd.DataFrame(
                    [[0.5, 0.75], [np.nan, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Minimum
            (
                "minimum",
                True,
                pd.DataFrame(
                    [[0.5, 0.75], [0.75, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Fill=True
            (
                "maximum",
                False,
                pd.DataFrame(
                    [[0.5, 1.0], [np.nan, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Maximum
            (
                "mean",
                False,
                pd.DataFrame(
                    [[0.5, 0.875], [np.nan, 0.0]],
                    columns="kinase1 kinase2".split(),
                    index="kinase1 kinase2".split(),
                ),
            ),  # Minimum
        ],
    )
    def test_get_kinase_distance_matrix(
        self, fingerprint_distance_generator, by, fill, structure_distance_matrix
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
        structure_distance_matrix : pandas.DataFrame
            xxx
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = (
            fingerprint_distance_generator.get_kinase_distance_matrix(by, fill)
        )

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)
