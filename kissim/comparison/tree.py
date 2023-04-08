"""
Process a KISSIM distance matrix into a clustered kissim-based tree with assignment of the 
mean distance (or mean similarity) to each branch.
The resulting tree is written to an output file in Newick format.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from opencadd.databases import klifs

logger = logging.getLogger(__name__)

# Problematic kinases
# - SgK495, a pseudokinase with incorrect annotation in KLIFS (will be resolved)
PROBLEMATIC_KINASES = ["SgK495"]


def from_file(
    distance_matrix_path,
    tree_path=None,
    annotation_path=None,
    clustering_method="ward",
    similarity=False,
):
    """
    Generate kissim-based kinase tree from file.
    Cluster kinases and save clusters in the Newick format.

    Parameters
    ----------
    distance_matrix_path : str or pathlib.Path
        Path to kissim kinase distance matrix (CSV file).
    tree_path : None or str or pathlib.Path
        Path to kinase tree file in Newick format (useful for FigTree).
        Recommended file suffix: .tree
    annotation_path : None or str or pathlib.Path
        Path to annotation CSV file containing kinase-group-family mappings (useful for FigTree).
    cmethod : str
        Clustering methods as offered by scipy, see [1]. Default is "ward", alternatives i.e.
        "complete", "weighted", "average", "centroid".
    similarity : bool
        If `True`, convert distance matrix into similarity matrix using
        `1 - distance_matrix / distance_matrix.max().max()`, else use distance matrix directly
        (default).

    Returns
    -------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    node_means : dict of int: float
        Mean distance or similarity (value) for each node index (key).

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """

    distance_matrix_path = Path(distance_matrix_path)

    # Read in KISSIM distance matrix from provided input file
    logger.info(f"Reading kinase matrix from {distance_matrix_path}")
    distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)

    # Generate tree
    tree, node_means = from_distance_matrix(
        distance_matrix, tree_path, annotation_path, clustering_method, similarity
    )

    return tree, node_means


def from_distance_matrix(
    distance_matrix,
    tree_path=None,
    annotation_path=None,
    clustering_method="ward",
    similarity=False,
):
    """
    Generate kissim-based kinase tree (cluster kinases and save clusters in the Newick format).

    Parameters
    ----------
    distance_matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    tree_path : None or str or pathlib.Path
        Path to kinase tree file in Newick format (useful for FigTree).
        Recommended file suffix: .tree
    annotation_path : None or str or pathlib.Path
        Path to annotation CSV file containing kinase-group-family mappings (useful for FigTree).
    cmethod : str
        Clustering methods as offered by scipy, see [1]. Default is "ward", alternatives i.e.
        "complete", "weighted", "average", "centroid".
    similarity : bool
        If `True`, convert distance matrix into similarity matrix using
        `1 - distance_matrix / distance_matrix.max().max()`, else use distance matrix directly
        (default).

    Returns
    -------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    node_means : dict of int: float
        Mean distance or similarity (value) for each node index (key).

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    """

    # Removing problematic entries if they exist
    for entry in PROBLEMATIC_KINASES:
        if entry in distance_matrix:
            distance_matrix.drop(entry, axis=0, inplace=True)
            distance_matrix.drop(entry, axis=1, inplace=True)

    # Curate diagonal - set to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # If matrix contains missing values, respective rows and columns must be dropped
    column_has_missing_values = distance_matrix.isna().any()
    column_names_with_missing_values = column_has_missing_values[column_has_missing_values].index
    distance_matrix = distance_matrix.drop(column_names_with_missing_values, axis=0).drop(
        column_names_with_missing_values, axis=1
    )

    # Hierarchical clustering
    logger.info(
        f"Clustering (method: {clustering_method}) and "
        f"calculating branch {'similarities' if similarity else 'distances'}"
    )
    hclust = sch.linkage(ssd.squareform(distance_matrix.values), method=clustering_method)
    logger.info("Converting clustering to a Newick tree")
    tree = sch.to_tree(hclust, False)

    # Calculate and assign mean distances - or mean similarities - for each of the branches
    if similarity:
        matrix = (1 - distance_matrix / distance_matrix.max().max()).copy()
        leaf_mean = 1
    else:
        matrix = distance_matrix.copy()
        leaf_mean = 0
    node_means = {}
    _get_node_means(tree, matrix, node_means, leaf_mean)

    # Optionally: Save tree and annotation to file
    if tree_path:
        _to_newick(tree, node_means, distance_matrix, tree_path)
    if annotation_path:
        _to_kinase_annotation(distance_matrix, annotation_path)

    return tree, node_means


def _to_newick(tree, node_means, distance_matrix, outputfile):
    """
    Save Newick tree to file.

    Parameters
    ----------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    node_means : dict of int: float
        Mean distance or similarity (value) for each node index (key).
    distance_matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    outputfile : str or pathlib.Path
        Path to kinase tree file (TREE file) in Newick format.
    """

    outputfile = Path(outputfile)

    logger.info(f"Writing resulting tree to {outputfile}")
    newick = ""
    newick = _get_newick(tree, newick, tree.dist, list(distance_matrix), node_means)
    with open(outputfile, "w") as f:
        f.write(newick)


def _to_kinase_annotation(distance_matrix, outputfile):
    """
    Save kinase annotations to file used for FigTree.

    Parameters
    ----------
    distance_matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    outputfile : str or pathlib.Path
        Path to kinase annotation file (CSV file) in FigTree format.
    """

    outputfile = Path(outputfile)

    logger.info(f"Writing resulting kinase annotation to {outputfile}")

    # Get kinase names from matrix
    kinase_names = distance_matrix.columns.to_list()

    # Query KLIFS for kinase details
    klifs_session = klifs.setup_remote()
    kinases = klifs_session.kinases.by_kinase_name(kinase_names)
    kinases = kinases[kinases["species.klifs"] == "Human"]
    kinases = kinases[["kinase.klifs_name", "kinase.family", "kinase.group"]]

    # Save to file
    kinases.to_csv(outputfile, sep="\t", index=False)


def _get_node_means(tree, matrix, node_means, leaf_mean):
    """
    Calculate and assign the mean of a tree clade based on their distance or similarity matrix.

    Parameters
    ----------
    tree : scipy.cluster.hierarchy.ClusterNode
        Clustering tree.
    matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    node_means : dict of int: float
        Mean distance (value) for each node index (key).
    leaf_mean : int
        Leaves technically do not have a mean; set identity value instead. In case `matrix` is a
        similarity matrix, set 1, in case it is a distance matrix, set 0.

    Returns
    -------
    node_means : dict of int: float
        Mean similarities (value) for each node index (key).
        Note: The return value is the populated input `node_means` object.
    """

    # Set mean for root node
    if tree.id not in node_means:
        # Get full matrix (root includes full dataset)
        matrix_all_clusters = matrix.values
        # Get matrix values from lower triangle without diagonal
        values_all_clusters = matrix_all_clusters[
            np.tril_indices(matrix_all_clusters.shape[0], -1)
        ]
        # Calculate mean of full dataset
        node_means[tree.id] = np.average(values_all_clusters)

    # Get mean only for nodes that are not a leaf
    if not tree.is_leaf():
        # Iterate over the nodes to the right and the left of the current node
        for left in [0, 1]:
            if left:
                # Get list of indices cluster left (A)
                indices = tree.get_left().pre_order(lambda x: x.id)
            else:
                # Get list of indices cluster right (B)
                indices = tree.get_right().pre_order(lambda x: x.id)

            if len(indices) > 1:
                # Get matrix subset for cluster members only
                matrix_by_cluster = matrix.iloc[indices, indices].values
                # Get matrix values from lower triangle without diagonal
                values_by_cluster = matrix_by_cluster[
                    np.tril_indices(matrix_by_cluster.shape[0], -1)
                ]
                # Calculate mean within a cluster
                if left:
                    node_means[tree.get_left().id] = np.average(values_by_cluster)
                    _get_node_means(tree.get_left(), matrix, node_means, leaf_mean)
                else:
                    node_means[tree.get_right().id] = np.average(values_by_cluster)
                    _get_node_means(tree.get_right(), matrix, node_means, leaf_mean)
            # Set leaf "mean" to identity (similarity=1; distance=0)
            else:
                if left:
                    node_means[tree.get_left().id] = leaf_mean
                else:
                    node_means[tree.get_right().id] = leaf_mean

    return node_means


def _get_newick(node, newick, parentdist, leaf_names, node_means):
    """
    Convert scipy Tree object into Newick string with annotated branches.

    Parameters
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    newick : str
        Newick string.
    parentdist : float
        Distance of parent node.
    leaf_names : list of str
        Leaf names (kinases).
    node_means : dict of int: float
        Mean distance (value) for each node index (key). Generated with `_get_mean_index`.

    Returns
    -------
    newick : str
        Newick string.
        Note: The return value is the populated input `newick` object.
    """

    if node.is_leaf():
        return f"{leaf_names[node.id]}:{round(parentdist - node.dist, 3)}{newick}"
    else:
        si_node = node_means[node.id]
        if len(newick) > 0:
            newick = f"){round(si_node, 3)}:{round(parentdist - node.dist, 3)}{newick}"
        else:
            newick = ");"
        newick = _get_newick(node.get_left(), newick, node.dist, leaf_names, node_means)
        newick = _get_newick(node.get_right(), f",{newick}", node.dist, leaf_names, node_means)
        newick = f"({newick}"
        return newick
