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
    "inputfile, outputfile, clustering_method, tree_string",
    [
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            "ward",
            "((ACTR2:0.261,AAK1:0.261)0.739:0.085,((ABL2:0.032,ABL1:0.032)0.968:0.132,ACK:0.165)0.894:0.182);",
        ),
        (
            PATH_TEST_DATA / "kinase_matrix.csv",
            PATH_TEST_DATA / "kissim_test.tree",
            "centroid",
            "((((ABL2:0.032,ABL1:0.032)0.968:0.11,ACK:0.143)0.894:0.092,AAK1:0.235)0.825:0.023,ACTR2:0.257);",
        ),
        (PATH_TEST_DATA / "kinase_matrix.csv", None, "ward", None),
    ],
)
def test_from_file(inputfile, outputfile, clustering_method, tree_string):

    with enter_temp_directory():

        tree_nodes, mean_similarity = tree.from_file(inputfile, outputfile, clustering_method)

        assert isinstance(tree_nodes, ClusterNode)
        assert isinstance(mean_similarity, dict)

        if outputfile:

            # Tree file there?
            tree_path = outputfile
            assert tree_path.exists()
            # Tree file correct?
            with open(tree_path, "r") as f:
                assert f.read() == tree_string

            # Annotation file there?
            annotation_path = outputfile.parent / "kinase_annotations.csv"
            assert annotation_path.exists()
