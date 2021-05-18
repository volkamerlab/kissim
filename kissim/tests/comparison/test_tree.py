"""
Unit and regression test for the kissim.comparison.tree module.
"""

from pathlib import Path

import pytest
import pandas as pd
from scipy.spatial import distance
from scipy.cluster import hierarchy

from kissim.utils import enter_temp_directory
from kissim.comparison import tree

# PATH_TEST_DATA = Path(__name__).parent / "kissim" / "tests" / "data"
PATH_TEST_DATA = Path(__file__).parent / "../data"


@pytest.mark.parametrize(
    "distance_matrix_path, tree_path, annotation_path, clustering_method, similarity, tree_string",
    [
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            PATH_TEST_DATA / "kinase_annotation.csv",
            "ward",
            True,
            "(((ACK:2.0,AAK1:2.0)0.6:3.431,(ACTR2:1.0,ABL1:1.0)0.8:4.431)0.367:0.322,ABL2:5.753);",
        ),
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            PATH_TEST_DATA / "kinase_annotation.csv",
            "ward",
            False,
            "(((ACK:2.0,AAK1:2.0)2.0:3.431,(ACTR2:1.0,ABL1:1.0)1.0:4.431)3.167:0.322,ABL2:5.753);",
        ),
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            None,
            "centroid",
            True,
            "(((ACK:2.0,AAK1:2.0)0.6:1.841,(ACTR2:1.0,ABL1:1.0)0.8:2.841)0.367:0.708,ABL2:4.548);",
        ),
        (PATH_TEST_DATA / "kinase_matrix.csv", None, None, "ward", True, None),
    ],
)
def test_from_file(
    distance_matrix_path, tree_path, annotation_path, clustering_method, similarity, tree_string
):

    with enter_temp_directory():

        tree_nodes, node_means = tree.from_file(
            distance_matrix_path, tree_path, annotation_path, clustering_method, similarity
        )

        assert isinstance(tree_nodes, hierarchy.ClusterNode)
        assert isinstance(node_means, dict)
        print(node_means)

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


@pytest.mark.parametrize(
    "distance_matrix, results, similarities",
    [
        (
            pd.DataFrame(
                [
                    [0, 4, 5, 2, 4],
                    [4, 0, 5, 4, 1],
                    [5, 5, 0, 5, 5],
                    [2, 4, 5, 0, 4],
                    [4, 1, 5, 4, 0],
                ]
            ),
            {8: 3.9, 7: 3.166667, 6: 2.0, 3: 0, 0: 0, 5: 1.0, 4: 0, 1: 0, 2: 0},
            False,
        ),
        (
            pd.DataFrame(
                [
                    [0, 4, 5, 2, 4],
                    [4, 0, 5, 4, 1],
                    [5, 5, 0, 5, 5],
                    [2, 4, 5, 0, 4],
                    [4, 1, 5, 4, 0],
                ]
            ),
            {8: 0.22, 7: 0.366667, 6: 0.6, 3: 1, 0: 1, 5: 0.8, 4: 1, 1: 1, 2: 1},
            True,
        ),
    ],
)
def test_get_node_means(distance_matrix, results, similarities):

    hclust = hierarchy.linkage(distance.squareform(distance_matrix), method="ward")
    cluster_node = hierarchy.to_tree(hclust, False)

    results_calculated = tree._get_node_means(cluster_node, distance_matrix, {}, similarities)
    assert results == pytest.approx(results_calculated, abs=1e-6)
