"""
Unit and regression test for the kissim.api.outliers module.
"""

from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.api import outliers
from kissim.encoding import FingerprintGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "fingerprints_path, distance_cutoff, fingerprints_wo_outliers_path, n_fingerprints_wo_outliers",
    [
        (
            PATH_TEST_DATA / "fingerprints_test.json",
            5,  # Low max > outliers!
            "fingerprints_clean_test.json.json",
            0,
        ),
        (
            PATH_TEST_DATA / "fingerprints_test.json",
            100,  # High max > no outliers!
            None,  # No output!
            2,
        ),
    ],
)
def test_outliers(
    fingerprints_path, distance_cutoff, fingerprints_wo_outliers_path, n_fingerprints_wo_outliers
):

    with enter_temp_directory():
        fingerprints = outliers(fingerprints_path, distance_cutoff, fingerprints_wo_outliers_path)
        assert isinstance(fingerprints, FingerprintGenerator)
        assert len(fingerprints.data) == n_fingerprints_wo_outliers

        if fingerprints_wo_outliers_path is not None:
            fingerprints_wo_outliers_path = Path(fingerprints_wo_outliers_path)
            assert fingerprints_wo_outliers_path.exists()
            assert (
                fingerprints_wo_outliers_path.parent / f"{fingerprints_wo_outliers_path.stem}.log"
            )
