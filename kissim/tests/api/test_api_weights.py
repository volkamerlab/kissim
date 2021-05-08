"""
Unit and regression test for the kissim.api.weights module.
"""

from pathlib import Path
import pytest

from kissim.utils import enter_temp_directory
from kissim.api import weights
from kissim.comparison import FingerprintDistanceGenerator

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "feature_distances_path, feature_weights, fingerprint_distances_path",
    [
        (
            (PATH_TEST_DATA / "feature_distances_test.csv").absolute(),
            None,
            "fingerprint_distances_test.csv",
        )
    ],
)
def test_weights(feature_distances_path, feature_weights, fingerprint_distances_path):

    with enter_temp_directory():
        fingerprint_distances = weights(
            feature_distances_path, feature_weights, fingerprint_distances_path
        )
        assert isinstance(fingerprint_distances, FingerprintDistanceGenerator)

        if fingerprint_distances_path is not None:
            fingerprint_distances_path = Path(fingerprint_distances_path)
            assert fingerprint_distances_path.exists()
            assert fingerprint_distances_path.parent / f"{fingerprint_distances_path.stem}.log"
