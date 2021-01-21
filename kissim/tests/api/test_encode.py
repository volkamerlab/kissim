"""
Unit and regression test for the kissim.api.encode module.
"""

from pathlib import Path

import pytest

from kissim.utils import enter_temp_directory
from kissim.api import encode

PATH_TEST_DATA = Path(__name__).parent / "kissim/tests/data/KLIFS_download"


@pytest.mark.parametrize(
    "structure_klifs_ids, json_filepath, n_cores, local_klifs_session",
    [
        ([12347], None, 1, None),
        ([12347], "fingerprints.json", 1, PATH_TEST_DATA),
    ],
)
def test_encode(structure_klifs_ids, json_filepath, n_cores, local_klifs_session):

    with enter_temp_directory():
        fingerprint_generator = encode(
            structure_klifs_ids, json_filepath, n_cores, local_klifs_session
        )
