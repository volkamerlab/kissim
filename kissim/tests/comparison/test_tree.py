"""
Unit and regression test for the kissim.comparison.tree module.
"""

from pathlib import Path

import pytest
from scipy.cluster.hierarchy import ClusterNode

from kissim.utils import enter_temp_directory
from kissim.comparison import tree

PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"


@pytest.mark.parametrize(
    "distance_matrix_path, tree_path, annotation_path, clustering_method, tree_string",
    [
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            PATH_TEST_DATA / "kinase_annotation.csv",
            "ward",
            "((ACTR2:0.261,AAK1:0.261)0.739:0.085,((ABL2:0.032,ABL1:0.032)0.968:0.132,ACK:0.165)0.894:0.182);",
        ),
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            None,
            "centroid",
            "((((ABL2:0.032,ABL1:0.032)0.968:0.11,ACK:0.143)0.894:0.092,AAK1:0.235)0.825:0.023,ACTR2:0.257);",
        ),
        (PATH_TEST_DATA / "kinase_matrix.csv", None, None, "ward", None),
    ],
)
def test_from_file(
    distance_matrix_path, tree_path, annotation_path, clustering_method, tree_string
):

    with enter_temp_directory():

        tree_nodes, mean_similarity = tree.from_file(
            distance_matrix_path, tree_path, annotation_path, clustering_method
        )

        assert isinstance(tree_nodes, ClusterNode)
        assert isinstance(mean_similarity, dict)

        if tree_path:

            tree_path = Path(tree_path)
            # Tree file there?
            assert tree_path.exists()
            # Tree file correct?
            with open(tree_path, "r") as f:
                assert f.read() == tree_string
            tree_path.unlink()

        if annotation_path:

            annotation_path = Path(annotation_path)
            # Annotation file there?
            assert annotation_path.exists()
            annotation_path.unlink()
