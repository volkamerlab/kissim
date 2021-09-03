"""
Unit and regression test for kissim.encoding.FingerprintNormalized.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local

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
        "values, fine_grained, values_normalized",
        [
            (None, False, None),
            (
                {
                    subpocket_name: [-100, -100, 100, 100]
                    for subpocket_name in ["hinge_region", "dfg_region", "front_pocket", "center"]
                },
                False,
                {
                    "hinge_region": [0.0, 0.0, 1.0, 1.0],
                    "dfg_region": [0.0, 0.0, 1.0, 1.0],
                    "front_pocket": [0.0, 0.0, 1.0, 1.0],
                    "center": [0.0, 0.0, 1.0, 1.0],
                },
            ),
        ],
    )
    def test_normalize_distances_bits(self, values, fine_grained, values_normalized):
        """
        Test normalization of distance bits.
        """

        fingerprint_normalized = FingerprintNormalized()
        values_normalized_calculated = fingerprint_normalized._normalize_distances_bits(
            values, fine_grained
        )
        assert values_normalized_calculated == values_normalized

    @pytest.mark.parametrize(
        "values, fine_grained, values_normalized",
        [
            (None, False, None),
            (
                {"test": [-100, -100, -100]},
                False,
                {"test": [0.0, 0.0, 0.0]},
            ),
            (
                {"test": [100, 100, 100]},
                False,
                {"test": [1.0, 1.0, 1.0]},
            ),
        ],
    )
    def test_normalize_moments_bits(self, values, fine_grained, values_normalized):
        """
        Test normalization of moments bits.
        """

        fingerprint_normalized = FingerprintNormalized()
        values_normalized_calculated = fingerprint_normalized._normalize_moments_bits(
            values, fine_grained
        )
        assert values_normalized_calculated == values_normalized
