"""
Unit and regression test for kissim.encoding.FingerprintGenerator.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.utils import enter_temp_directory
from kissim.encoding import FingerprintGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")


class TestFingerprintGenerator:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "structure_klifs_ids, klifs_session, n_cores, fingerprints_values_array_sum",
        [
            ([109, 110, 118], REMOTE, 1, 14627.0178),
            ([109, 110, 118], REMOTE, 2, 14627.0178),
            ([109, 110, 118], LOCAL, 1, 14627.0178),
            ([109, 110, 118], LOCAL, 2, 14627.0178),
        ],
    )
    def test_from_structure_klifs_id(
        self, structure_klifs_ids, klifs_session, n_cores, fingerprints_values_array_sum
    ):
        """
        Test if fingerprints can be generated locally and remotely in sequence and in parallel.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores
        )
        fingerprints_values_array_sum_calculated = sum(
            [
                np.nansum(fingerprint.values_array(True, True, True))
                for structure_klifs_id, fingerprint in fingerprints.data.items()
            ]
        )
        assert (
            pytest.approx(fingerprints_values_array_sum_calculated, abs=1e-4)
            == fingerprints_values_array_sum
        )

    @pytest.mark.parametrize(
        "structure_klifs_ids, values_array_sum",
        [([109, 110, 118], 14627.0178)],
    )
    def test_to_from_json(self, structure_klifs_ids, values_array_sum):
        """
        Test if saving/loading a fingerprint to/from a json file.
        """

        fingerprints = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, LOCAL, 1
        )
        json_filepath = Path("fingerprints.json")

        with enter_temp_directory():

            # Save json file
            fingerprints.to_json(json_filepath)
            assert json_filepath.exists()

            # Load json file
            fingerprints_reloaded = FingerprintGenerator.from_json(json_filepath)

        assert isinstance(fingerprints_reloaded, FingerprintGenerator)
        # Attribute data
        assert list(fingerprints.data.keys()) == list(fingerprints_reloaded.data.keys())
        values_array_sum_calculated = sum(
            [
                np.nansum(fingerprint.values_array(True, True, True))
                for structure_klifs_id, fingerprint in fingerprints_reloaded.data.items()
            ]
        )
        assert pytest.approx(values_array_sum_calculated, abs=1e-4) == values_array_sum
