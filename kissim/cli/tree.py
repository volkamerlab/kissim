"""kissim_to_newick

This is a small tool for processing a KISSIM similarity matrix into a clustered
kissim-based tree with assignment of the mean similarity to each branch.
The resulting tree is written to an output file in Newick format.
"""

from kissim.api import tree

def tree_from_cli(args):
    """
    Generate kissim-based kinase tree.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    tree(args.input, args.output)
