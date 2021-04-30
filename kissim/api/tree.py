"""
kissim.api.tree

Main API for kissim tree generation.
"""

from kissim.comparison import tree


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

    tree(inputfile, outputfile)
