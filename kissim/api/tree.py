"""
kissim.api.tree

Main API for kissim tree generation.
"""

from kissim.comparison import kissim_to_newick


def tree(inputfile, outputfile):
    """
    Generate kissim-based kinase tree.

    Parameters
    ----------
    inputfile : str or pathlib.Path
        Path to kissim kinase matrix (CSV file).
    outputfile : str or pathlib.Path
        Path to kinase tree file (TREE file) in Newick format.
    """

    kissim_to_newick(inputfile, outputfile)
