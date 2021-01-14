"""
Unit and regression test for kissim.encoding.fingerprint.Fingerprint and its
parent class kissim.encoding.base.FingerprintBase.
"""

from pathlib import Path
import pytest

import numpy as np
from opencadd.databases.klifs import setup_local, setup_remote

from kissim.encoding import Fingerprint

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
REMOTE = setup_remote()
LOCAL = setup_local(PATH_TEST_DATA / "KLIFS_download")
print(LOCAL._database)


class TestFingerprint:
    """
    Test common functionalities in the PocketBioPython and PocketDataFrame classes.
    """

    @pytest.mark.parametrize(
        "structure_klifs_id",
        [109, 110, 118],
    )
    def test_from_structure_klifs_id(self, structure_klifs_id):
        """
        Test if Fingerprint can be set locally and remotely.
        """

        fingerprint1 = Fingerprint.from_structure_klifs_id(structure_klifs_id, LOCAL)
        fingerprint2 = Fingerprint.from_structure_klifs_id(structure_klifs_id, REMOTE)

        v1 = fingerprint1.values_array()
        v2 = fingerprint2.values_array()
        assert np.allclose(v1, v2, rtol=0, atol=0, equal_nan=True)
