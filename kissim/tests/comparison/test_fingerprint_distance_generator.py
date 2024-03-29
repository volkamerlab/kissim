"""
Unit and regression test for the kissim.comparison.FingerprintDistanceGenerator class.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.comparison import FingerprintDistanceGenerator


PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestsFingerprintDistanceGenerator:
    """
    Test FingerprintDistanceGenerator class methods.
    """

    @pytest.mark.parametrize(
        "feature_weights, structure_kinase_ids, structure_pair_ids, kinase_pair_ids, structure_ids, kinase_ids",
        [
            (
                None,
                [("pdbA", "kinaseA"), ("pdbB", "kinaseA"), ("pdbC", "kinaseB")],
                [["pdbA", "pdbB"], ["pdbA", "pdbC"], ["pdbB", "pdbC"]],
                [["kinaseA", "kinaseA"], ["kinaseA", "kinaseB"], ["kinaseA", "kinaseB"]],
                ["pdbA", "pdbB", "pdbC"],
                ["kinaseA", "kinaseB"],
            )
        ],
    )
    def test_from_feature_distances_generator(
        self,
        feature_distances_generator,
        feature_weights,
        structure_kinase_ids,
        structure_pair_ids,
        kinase_pair_ids,
        structure_ids,
        kinase_ids,
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
        assert isinstance(fingerprint_distance_generator.data, pd.DataFrame)
        data_columns = [
            "structure.1",
            "structure.2",
            "kinase.1",
            "kinase.2",
            "distance",
            "bit_coverage",
        ]
        assert fingerprint_distance_generator.data.columns.to_list() == data_columns
        assert fingerprint_distance_generator.structure_kinase_ids == structure_kinase_ids

        # Test properties
        assert fingerprint_distance_generator.structure_pair_ids == structure_pair_ids
        assert fingerprint_distance_generator.kinase_pair_ids == kinase_pair_ids
        assert fingerprint_distance_generator.structure_ids == structure_ids
        assert fingerprint_distance_generator.kinase_ids == kinase_ids
        assert isinstance(fingerprint_distance_generator.distances, np.ndarray)
        assert isinstance(fingerprint_distance_generator.bit_coverages, np.ndarray)

    @pytest.mark.parametrize(
        "structure_klifs_ids, klifs_session, feature_weights",
        [
            ([110, 118], REMOTE, None),
            ([110, 118], REMOTE, None),
            ([110, 118], LOCAL, None),
            ([110, 118], LOCAL, None),
            ([110, 118], None, None),
        ],
    )
    def test_from_structure_klifs_ids(self, structure_klifs_ids, klifs_session, feature_weights):
        """
        Test FingerprintDistanceGenerator generation from structure KLIFS IDs.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, feature_weights
        )
        assert isinstance(fingerprint_distance_generator, FingerprintDistanceGenerator)

    def test_from_fingerprint_generator(self, fingerprint_generator):
        """
        Test FingerprintDistanceGenerator generation from a fingerprint generator.
        """

        fingerprint_distance_generator = FingerprintDistanceGenerator.from_fingerprint_generator(
            fingerprint_generator
        )
        assert isinstance(fingerprint_distance_generator, FingerprintDistanceGenerator)

    @pytest.mark.parametrize(
        "coverage_min, structure_distance_matrix",
        [
            (
                0.0,
                pd.DataFrame(
                    [[0.0, 0.75, 1.0], [0.75, 0.0, 0.8], [1.0, 0.8, 0.0]],
                    columns="pdbA pdbB pdbC".split(),
                    index="pdbA pdbB pdbC".split(),
                ),
            ),
            (
                0.9,
                pd.DataFrame(
                    [[0.0, 0.75, 1.0], [0.75, 0.0, np.nan], [1.0, np.nan, 0.0]],
                    columns="pdbA pdbB pdbC".split(),
                    index="pdbA pdbB pdbC".split(),
                ),
            ),  # Coverage minimum set
        ],
    )
    def test_structure_distance_matrix(
        self, fingerprint_distance_generator, coverage_min, structure_distance_matrix
    ):
        """
        Test if structure distance matrix is correct.

        Parameters
        ----------
        fingerprint_distance_generator : FingerprintDistanceGenerator
            Fingerprint distance for multiple fingerprint pairs.
        coverage_min : float
            Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
            coverage restrictions).
        structure_distance_matrix : pandas.DataFrame
            Structure distance matrix.
        """

        # Test generation of structure distance matrix
        structure_distance_matrix_calculated = (
            fingerprint_distance_generator.structure_distance_matrix(coverage_min)
        )
        structure_distance_matrix.columns.name = "structure.2"
        structure_distance_matrix.index.name = "structure.1"

        assert structure_distance_matrix_calculated.equals(structure_distance_matrix)

    @pytest.mark.parametrize(
        "by, fill_diagonal, coverage_min, kinase_distance_matrix",
        [
            (
                "minimum",
                False,
                0.0,
                pd.DataFrame(
                    [[0.75, 0.8], [0.8, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Minimum
            (
                "minimum",
                True,
                0.0,
                pd.DataFrame(
                    [[0, 0.8], [0.8, 0]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Minimum and fill diagonal
            (
                "minimum",
                False,
                0.9,
                pd.DataFrame(
                    [[0.75, 1.0], [1.0, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Minimum and coverage
            (
                "maximum",
                False,
                0.0,
                pd.DataFrame(
                    [[0.75, 1.0], [1.0, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Maximum
            (
                "mean",
                False,
                0.0,
                pd.DataFrame(
                    [[0.75, 0.9], [0.9, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Mean
            (
                "median",
                False,
                0.0,
                pd.DataFrame(
                    [[0.75, 0.9], [0.9, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Median
            (
                "size",
                False,
                0.0,
                pd.DataFrame(
                    [[1, 2], [2, 0]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Size
            (
                "size",
                True,
                0.0,
                pd.DataFrame(
                    [[1, 2], [2, 0]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Size
            (
                "std",
                False,
                0.0,
                pd.DataFrame(
                    [[np.nan, 0.141], [0.141, np.nan]],
                    columns=["kinaseA", "kinaseB"],
                    index=["kinaseA", "kinaseB"],
                ),
            ),  # Std
        ],
    )
    def test_kinase_distance_matrix(
        self,
        fingerprint_distance_generator,
        by,
        fill_diagonal,
        coverage_min,
        kinase_distance_matrix,
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
        coverage_min : float
            Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
            coverage restrictions).
        kinase_distance_matrix : pandas.DataFrame
            Kinase distance matrix.
        """

        # Set index and column names for template kinase distance matrix
        kinase_distance_matrix.index.name = "kinase.1"
        kinase_distance_matrix.columns.name = "kinase.2"

        # Test generation of structure distance matrix
        kinase_distance_matrix_calculated = fingerprint_distance_generator.kinase_distance_matrix(
            by, fill_diagonal, coverage_min
        )

        assert kinase_distance_matrix_calculated.equals(kinase_distance_matrix)
