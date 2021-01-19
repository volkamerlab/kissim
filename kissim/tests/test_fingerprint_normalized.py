"""
Unit and regression test for kissim.encoding.FingerprintNormalized.
"""

import pytest

import numpy as np
from opencadd.databases.klifs import setup_remote

from kissim.definitions import DISTANCE_CUTOFFS, MOMENT_CUTOFFS
from kissim.encoding import Fingerprint, FingerprintNormalized

REMOTE = setup_remote()


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

        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)
        fingerprint_normalized = FingerprintNormalized.from_fingerprint(fingerprint)

    @pytest.mark.parametrize(
        "values, values_normalized",
        [
            (
                {
                    "size": [0, 1, 3, 4],
                    "hbd": [-1, 0, 3, 4],
                    "hba": [-1, 0, 2, 3],
                    "charge": [-2, -1, 1, 2],
                    "aromatic": [-1, 0, 1, 2],
                    "aliphatic": [-1, 0, 1, 2],
                    "sco": [-1, 0, 2, 3],
                    "exposure": [-1, 0, 1, 2],
                },
                {
                    "size": [0.0, 0.0, 1.0, 1.0],
                    "hbd": [0.0, 0.0, 1.0, 1.0],
                    "hba": [0.0, 0.0, 1.0, 1.0],
                    "charge": [0.0, 0.0, 1.0, 1.0],
                    "aromatic": [0.0, 0.0, 1.0, 1.0],
                    "aliphatic": [0.0, 0.0, 1.0, 1.0],
                    "sco": [0.0, 0.0, 1.0, 1.0],
                    "exposure": [0.0, 0.0, 1.0, 1.0],
                },
            )
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
        print(values_normalized_calculated)
        assert values_normalized_calculated == values_normalized

    @pytest.mark.parametrize(
        "values, values_normalized",
        [
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
            )
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
