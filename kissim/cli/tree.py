"""
kissim.cli.tree

Process a KISSIM distance matrix into a clustered kissim-based tree with assignment of the 
mean similarity to each branch.
The resulting tree is written to an output file in Newick format.
"""

from kissim.comparison import tree


def tree_from_cli(args):
    """
    Generate kissim-based kinase tree.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    tree.from_file(args.input, args.output, args.annotation, args.clustering)
