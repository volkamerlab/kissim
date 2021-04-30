"""kissim_to_newick

This is a small tool for processing a KISSIM similarity matrix into a clustered
kissim-based tree with assignment of the mean similarity to each branch.
The resulting tree is written to an output file in Newick format.
"""

import numpy as np
import pandas as pd
import sys
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

def kissim_to_newick(inputfile, outputfile):
    """Main function for the kissim_to_newick tool."""
    print("\033[1mkissim_to_newick - converting kissim similarities to a Newick tree\033[0m\n---")

    # Read in KISSIM similarity matrix from provided inputfile
    print("Reading KISSIM data from {}".format(inputfile))
    distance_matrix = pd.read_csv(inputfile, index_col=0)

    # Removing problematic entries if they exist
    # Removal of SgK495, a pseudokinase with incorrect annotation in KLIFS (will be resolved)
    problematic_entries = ["SgK495"]
    for entry in problematic_entries:
        if entry in distance_matrix:
            distance_matrix.drop(entry, axis=0, inplace=True)
            distance_matrix.drop(entry, axis=1, inplace=True)

    # Curate diagonal - set to 0
    np.fill_diagonal(distance_matrix.values, 0)

    # Hierarchical clustering (Ward by default)
    # Alternatives: 'complete', 'weighted', 'average', 'centroid'
    print("Clustering and calculating branch similarities")
    cmethod = "ward"
    hclust = sch.linkage(ssd.squareform(distance_matrix.values), method=cmethod)
    tree = sch.to_tree(hclust, False)

    # Calculate and assign mean similarity for each of the branches
    mean_similarity = {}
    getMeanIndex(tree, distance_matrix, mean_similarity)

    # Output in Newick format
    print("Writing resulting tree to {}".format(outputfile))
    newick = getNewick(tree, "", tree.dist, list(distance_matrix), mean_similarity)
    tree_file = open(outputfile, "w")
    tree_file.write(newick)
    tree_file.close()

    # Done
    print("\033[0;31mDone!\033[0m")


def getMeanIndex(node, distance_matrix, results):
    """Method for calculating an assiging the mean similarity for tree branches."""
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
                    getMeanIndex(node.get_left(), distance_matrix, results)
                else:
                    results[node.get_right().id] = np.average(si)
                    getMeanIndex(node.get_right(), distance_matrix, results)
            else:
                results[node.get_left().id] = 1


def getNewick(node, newick, parentdist, leaf_names, mean_similarity):
    """Method for converting scipy Tree object into Newick string with annotated branches."""
    if node.is_leaf():
        return "%s:%.3f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        si_node = mean_similarity[node.id]
        if len(newick) > 0:
            newick = ")%.3f:%.3f%s" % (si_node, parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names, mean_similarity)
        newick = getNewick(
            node.get_right(), ",%s" % (newick), node.dist, leaf_names, mean_similarity
        )
        newick = "(%s" % (newick)
        return newick
