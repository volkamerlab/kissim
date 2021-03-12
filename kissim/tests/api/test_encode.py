"""
Unit and regression test for the kissim.api.encode module.
"""

from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.api import encode
from kissim.encoding import FingerprintGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "structure_klifs_ids, fingerprints_json_filepath, n_cores, local_klifs_download_path",
    [
        ([12347, 110], None, 1, None),
        ([12347, 110], "fingerprints.json", 1, PATH_TEST_DATA / "KLIFS_download"),
        ([12347, 110], None, 2, PATH_TEST_DATA / "KLIFS_download"),
    ],
)
def test_encode(
    structure_klifs_ids, fingerprints_json_filepath, n_cores, local_klifs_download_path
):

    fingerprint_generator = encode(
        structure_klifs_ids, fingerprints_json_filepath, n_cores, local_klifs_download_path
    )
    assert isinstance(fingerprint_generator, FingerprintGenerator)

    if fingerprints_json_filepath is not None:
        fingerprints_json_filepath = Path(fingerprints_json_filepath)
        assert fingerprints_json_filepath.exists()
        fingerprints_json_filepath.unlink()
