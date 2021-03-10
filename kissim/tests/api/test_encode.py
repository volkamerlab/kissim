"""
Unit and regression test for the kissim.api.encode module.
"""

from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.api import encode
from kissim.encoding import FingerprintGenerator


@pytest.mark.parametrize(
    "structure_klifs_ids, json_filepath, n_cores, local_klifs_session",
    [
        ([110], None, 1, None),
        ([110], "fingerprints.json", 1, None),
        ([110], None, 2, None),
    ],
)
def test_encode(structure_klifs_ids, json_filepath, n_cores, local_klifs_session):

    with enter_temp_directory():
        fingerprint_generator = encode(
            structure_klifs_ids, json_filepath, n_cores, local_klifs_session
        )
        assert isinstance(fingerprint_generator, FingerprintGenerator)

        if json_filepath is not None:
            assert Path(json_filepath).exists()
