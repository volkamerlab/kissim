"""
Process a KISSIM distance matrix into a clustered kissim-based tree with assignment of the 
mean similarity to each branch.
The resulting tree is written to an output file in Newick format.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from opencadd.databases import klifs

# Problematic kinases
# - SgK495, a pseudokinase with incorrect annotation in KLIFS (will be resolved)
PROBLEMATIC_KINASES = ["SgK495"]


def from_file(
    distance_matrix_path, tree_path=None, annotation_path=None, clustering_method="ward"
):
    """
    Generate kissim-based kinase tree from file.
    Cluster kinases and save clusters in the Newick format.

    Parameters
    ----------
    distance_matrix_path : str or pathlib.Path
        Path to kissim kinase matrix (CSV file).
    tree_path : None or str or pathlib.Path
        Path to kinase tree file in Newick format (useful for FigTree).
        Recommended file suffix: .tree
    annotation_path : None or str or pathlib.Path
        Path to annotation CSV file containing kinase-group-family mappings (useful for FigTree).
    cmethod : str
        Clustering methods as offered by scipy, see [1]. Default is "ward", alternatives i.e.
        "complete", "weighted", "average", "centroid".

    Returns
    -------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    mean_similarity : dict of int: float
        Mean similarity (value) for each node index (key).
    """

    distance_matrix_path = Path(distance_matrix_path)

    # Read in KISSIM distance matrix from provided input file
    print(f"Reading kinase matrix from {distance_matrix_path}")
    distance_matrix = pd.read_csv(distance_matrix_path, index_col=0)

    # Generate tree
    tree, mean_similarity = from_distance_matrix(
        distance_matrix, tree_path, annotation_path, clustering_method
    )

    return tree, mean_similarity


def from_distance_matrix(
    distance_matrix, tree_path=None, annotation_path=None, clustering_method="ward"
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

    Returns
    -------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    mean_similarity : dict of int: float
        Mean similarity (value) for each node index (key).

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

    # Hierarchical clustering (Ward by default)
    # Alternatives: 'complete', 'weighted', 'average', 'centroid'
    print(f"Clustering (method: {clustering_method}) and calculating branch similarities")
    hclust = sch.linkage(ssd.squareform(distance_matrix.values), method=clustering_method)
    print("Converting clustering to a Newick tree")
    tree = sch.to_tree(hclust, False)

    # Calculate and assign mean similarity for each of the branches
    mean_similarity = {}
    _get_mean_index(tree, distance_matrix, mean_similarity)

    # Optionally: Save tree and annotation to file
    if tree_path:
        _to_newick(tree, mean_similarity, distance_matrix, tree_path)
    if annotation_path:
        _to_kinase_annotation(distance_matrix, annotation_path)

    return tree, mean_similarity


def _to_newick(tree, mean_similarity, distance_matrix, outputfile):
    """
    Save Newick tree to file.

    Parameters
    ----------
    tree : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    mean_similarity : dict of int: float
        Mean similarity (value) for each node index (key).
    distance_matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    outputfile : str or pathlib.Path
        Path to kinase tree file (TREE file) in Newick format.
    """

    outputfile = Path(outputfile)

    print(f"Writing resulting tree to {outputfile}")
    newick = ""
    newick = _get_newick(tree, newick, tree.dist, list(distance_matrix), mean_similarity)
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

    print(f"Writing resulting kinase annotation to {outputfile}")

    # Get kinase names from matrix
    kinase_names = distance_matrix.columns.to_list()

    # Query KLIFS for kinase details
    klifs_session = klifs.setup_remote()
    kinases = klifs_session.kinases.by_kinase_name(kinase_names)
    kinases = kinases[kinases["species.klifs"] == "Human"]
    kinases = kinases[["kinase.klifs_name", "kinase.family", "kinase.group"]]

    # Save to file
    kinases.to_csv(outputfile, sep="\t", index=False)


def _get_mean_index(node, distance_matrix, results):
    """
    Calculating and assign the mean similarity for tree branches.

    Parameters
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        Cluster node.
    distance_matrix : pandas.DataFrame
        Distance matrix on which clustering is based.
    results : dict of int: float
        Mean distance (value) for each node index (key).

    Returns
    -------
    results : dict of int: float
        Mean distance (value) for each node index (key).
        Note: The return value is the populated input `results` object.
    """

    if node.id not in results:
        results[node.id] = 1
    if not node.is_leaf():
        for left in range(0, 2):
            if left:
                # get list of indices cluster left
                indices = node.get_left().pre_order(lambda x: x.id)
            else:
                # get list of indices cluster right (B)
                indices = node.get_right().pre_order(lambda x: x.id)

            if len(indices) > 1:
                # calculate mean similarity - cluster A
                similarity_cluster = (1 - distance_matrix.iloc[indices, indices]).values
                si = similarity_cluster[np.tril_indices(similarity_cluster.shape[0], -1)]
                if left:
                    results[node.get_left().id] = np.average(si)
                    _get_mean_index(node.get_left(), distance_matrix, results)
                else:
                    results[node.get_right().id] = np.average(si)
                    _get_mean_index(node.get_right(), distance_matrix, results)
            else:
                results[node.get_left().id] = 1


def _get_newick(node, newick, parentdist, leaf_names, mean_similarity):
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
    mean_similarity : dict of int: float
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
        si_node = mean_similarity[node.id]
        if len(newick) > 0:
            newick = f"){round(si_node, 3)}:{round(parentdist - node.dist, 3)}{newick}"
        else:
            newick = ");"
        newick = _get_newick(node.get_left(), newick, node.dist, leaf_names, mean_similarity)
        newick = _get_newick(
            node.get_right(), f",{newick}", node.dist, leaf_names, mean_similarity
        )
        newick = f"({newick}"
        return newick
