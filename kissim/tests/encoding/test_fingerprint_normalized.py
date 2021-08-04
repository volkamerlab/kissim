"""
Unit and regression test for kissim.encoding.FingerprintNormalized.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local

from kissim.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from kissim.encoding import Fingerprint, FingerprintNormalized

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestFingerprintNormalized:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109, 12347],
    )
    def test_fingerprint(self, structure_klifs_id):
        """
        Test if normalized fingerprint can be generated from fingerprint.
        Includes tests for the method _normalize() used in the class __init__ method.
        """

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, LOCAL)
        fingerprint_normalized = FingerprintNormalized.from_fingerprint(fingerprint)
        assert isinstance(fingerprint_normalized, FingerprintNormalized)
        assert fingerprint.structure_klifs_id == fingerprint_normalized.structure_klifs_id
        assert fingerprint.kinase_name == fingerprint_normalized.kinase_name
        assert fingerprint.residue_ids == fingerprint_normalized.residue_ids
        assert fingerprint.residue_ixs == fingerprint_normalized.residue_ixs
        # Test the only remaining attribute `values_dict` below!

    @pytest.mark.parametrize(
        "values, values_normalized",
        [
            (None, None),
            (
                {
                    "size": [0, 1, 3, np.nan],
                    "hbd": [-1, 0, 3, 4],
                    "hba": [-1, 0, 2, 3],
                    "charge": [-2, -1, 1, 2],
                    "aromatic": [-1, 0, 1, 2],
                    "aliphatic": [-1, 0, 1, 2],
                    "sco": [0, 1, 2, 3, 4],
                    "exposure": [0, 1, 2, 3, 4],
                },
                {
                    "size": [0.0, 0.0, 1.0, np.nan],
                    "hbd": [0.0, 0.0, 1.0, 1.0],
                    "hba": [0.0, 0.0, 1.0, 1.0],
                    "charge": [0.0, 0.0, 1.0, 1.0],
                    "aromatic": [0.0, 0.0, 1.0, 1.0],
                    "aliphatic": [0.0, 0.0, 1.0, 1.0],
                    "sco": [0.0, 0.0, 0.5, 1.0, 1.0],
                    "exposure": [0.0, 0.0, 0.5, 1.0, 1.0],
                },
            ),
        ],
    )
    def test_normalize_physicochemical_bits(self, values, values_normalized):
        """
        Test normalization of physicochemcial bits.
        """

        fingerprint_normalized = FingerprintNormalized()
        values_normalized_calculated = fingerprint_normalized._normalize_physicochemical_bits(
            values
        )
        assert values_normalized_calculated == values_normalized

    @pytest.mark.parametrize(
        "values, values_normalized",
        [
            (None, None),
            (
                {
                    "hinge_region": [
                        DISTANCE_CUTOFFS["hinge_region"][0] - 1,
                        DISTANCE_CUTOFFS["hinge_region"][0],
                        DISTANCE_CUTOFFS["hinge_region"][1],
                        DISTANCE_CUTOFFS["hinge_region"][1] + 1,
                    ],
                    "dfg_region": [
                        DISTANCE_CUTOFFS["dfg_region"][0] - 1,
                        DISTANCE_CUTOFFS["dfg_region"][0],
                        DISTANCE_CUTOFFS["dfg_region"][1],
                        DISTANCE_CUTOFFS["dfg_region"][1] + 1,
                    ],
                    "front_pocket": [
                        DISTANCE_CUTOFFS["front_pocket"][0] - 1,
                        DISTANCE_CUTOFFS["front_pocket"][0],
                        DISTANCE_CUTOFFS["front_pocket"][1],
                        DISTANCE_CUTOFFS["front_pocket"][1] + 1,
                    ],
                    "center": [
                        DISTANCE_CUTOFFS["center"][0] - 1,
                        DISTANCE_CUTOFFS["center"][0],
                        DISTANCE_CUTOFFS["center"][1],
                        DISTANCE_CUTOFFS["center"][1] + 1,
                    ],
                },
                {
                    "hinge_region": [0.0, 0.0, 1.0, 1.0],
                    "dfg_region": [0.0, 0.0, 1.0, 1.0],
                    "front_pocket": [0.0, 0.0, 1.0, 1.0],
                    "center": [0.0, 0.0, 1.0, 1.0],
                },
            ),
        ],
    )
    def test_normalize_distances_bits(self, values, values_normalized):
        """
        Test normalization of distance bits.
        """

        fingerprint_normalized = FingerprintNormalized()
        values_normalized_calculated = fingerprint_normalized._normalize_distances_bits(values)
        assert values_normalized_calculated == values_normalized

    @pytest.mark.parametrize(
        "values, values_normalized",
        [
            (None, None),
            (
                {
                    "test": [
                        MOMENT_CUTOFFS[1][0] - 1,
                        MOMENT_CUTOFFS[2][0] - 1,
                        MOMENT_CUTOFFS[3][0] - 1,
                    ],
                },
                {
                    "test": [0.0, 0.0, 0.0],
                },
            ),
            (
                {
                    "test": [
                        MOMENT_CUTOFFS[1][0],
                        MOMENT_CUTOFFS[2][0],
                        MOMENT_CUTOFFS[3][0],
                    ],
                },
                {
                    "test": [0.0, 0.0, 0.0],
                },
            ),
            (
                {
                    "test": [
                        MOMENT_CUTOFFS[1][1],
                        MOMENT_CUTOFFS[2][1],
                        MOMENT_CUTOFFS[3][1],
                    ],
                },
                {
                    "test": [1.0, 1.0, 1.0],
                },
            ),
            (
                {
                    "test": [
                        MOMENT_CUTOFFS[1][1] + 1,
                        MOMENT_CUTOFFS[2][1] + 1,
                        MOMENT_CUTOFFS[3][1] + 1,
                    ],
                },
                {
                    "test": [1.0, 1.0, 1.0],
                },
            ),
        ],
    )
    def test_normalize_moments_bits(self, values, values_normalized):
        """
        Test normalization of moments bits.
        """

        fingerprint_normalized = FingerprintNormalized()
        values_normalized_calculated = fingerprint_normalized._normalize_moments_bits(values)
        assert values_normalized_calculated == values_normalized

    @pytest.mark.parametrize(
        "value, minimum, maximum, value_normalized",
        [(15, 10, 20, 0.5), (10, 10, 20, 0.0), (0, 10, 20, 0.0), (np.nan, 10, 20, np.nan)],
    )
    def test_min_max_normalization(self, value, minimum, maximum, value_normalized):
        """
        Test min-max normalization
        """

        fingerprint_normalized = FingerprintNormalized()
        value_normalized_calculated = fingerprint_normalized._min_max_normalization(
            value, minimum, maximum
        )
        if not np.isnan(value):
            assert pytest.approx(value_normalized_calculated, abs=1e-4) == value_normalized
