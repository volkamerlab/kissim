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

        feature_distances_filepath = output_path / "feature_distances.csv"
        assert feature_distances_filepath.exists()
        feature_distances_filepath.unlink()

        fingerprint_distance_filepath = output_path / "fingerprint_distances.csv"
        assert fingerprint_distance_filepath.exists()
        fingerprint_distance_filepath.unlink()

        kinase_matrix_filepath = output_path / "fingerprint_distances_to_kinase_matrix.csv"
        assert kinase_matrix_filepath.exists()
        kinase_matrix_filepath.unlink()

        kinase_tree_filepath = output_path / "fingerprint_distances_to_kinase_clusters.tree"
        assert kinase_tree_filepath.exists()
        kinase_tree_filepath.unlink()

        kinase_annotation_filepath = output_path / "kinase_annotation.csv"
        assert kinase_annotation_filepath.exists()
        kinase_annotation_filepath.unlink()
