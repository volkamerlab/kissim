"""
Unit and regression test for the kissim.api.compare module.
"""

from pathlib import Path

import pytest

from kissim.api import encode, compare
from kissim.encoding import FingerprintGenerator


@pytest.mark.parametrize(
    "output_path, feature_weights, n_cores",
    [
        (None, None, 1),
        (None, None, 2),
        (None, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
        (".", [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1),
    ],
)
def test_compare(fingerprint_generator, output_path, feature_weights, n_cores):

    compare(fingerprint_generator, output_path, feature_weights, n_cores)

    if output_path is not None:
        output_path = Path(output_path)

        feature_distances_json_filepath = output_path / "feature_distances.json"
        assert feature_distances_json_filepath.exists()
        feature_distances_json_filepath.unlink()

        fingerprint_distance_json_filepath = output_path / "fingerprint_distances.json"
        assert fingerprint_distance_json_filepath.exists()
        fingerprint_distance_json_filepath.unlink()