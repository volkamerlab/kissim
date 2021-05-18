"""
kissim.cli.tree

Process a KISSIM distance matrix into a clustered kissim-based tree with assignment of the 
mean distance (or mean similarity) to each branch.
The resulting tree is written to an output file in Newick format.
"""

from kissim.comparison import tree
from kissim.cli.utils import configure_logger


def tree_from_cli(args):
    """
    Generate kissim-based kinase tree.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger()
    tree.from_file(args.input, args.output, args.annotation, args.clustering)
