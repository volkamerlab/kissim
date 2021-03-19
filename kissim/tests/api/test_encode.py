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
    "structure_klifs_ids, fingerprints_json_filepath, local_klifs_download_path, n_cores",
    [
        ([12347, 110], None, None, 1),
        ([12347, 110], "fingerprints.json", PATH_TEST_DATA / "KLIFS_download", 1),
        ([12347, 110], None, PATH_TEST_DATA / "KLIFS_download", 2),
    ],
)
def test_encode(
    structure_klifs_ids, fingerprints_json_filepath, local_klifs_download_path, n_cores
):

    fingerprint_generator = encode(
        structure_klifs_ids, fingerprints_json_filepath, local_klifs_download_path, n_cores
    )
    assert isinstance(fingerprint_generator, FingerprintGenerator)

    if fingerprints_json_filepath is not None:
        fingerprints_json_filepath = Path(fingerprints_json_filepath)
        assert fingerprints_json_filepath.exists()
        fingerprints_json_filepath.unlink()
