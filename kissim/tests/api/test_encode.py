"""
Unit and regression test for the kissim.api.encode module.
"""

import pytest

from kissim.utils import enter_temp_directory
from kissim.api import encode


@pytest.mark.parametrize(
    "structure_klifs_ids, json_filepath, n_cores, local_klifs_session",
    [([110], None, 1, None)],
)
def test_encode_bla(structure_klifs_ids, json_filepath, n_cores, local_klifs_session):

    with enter_temp_directory():
        fingerprint_generator = encode(
            structure_klifs_ids, json_filepath, n_cores, local_klifs_session
        )
