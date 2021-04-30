"""
kissim.api.tree

Main API for kissim tree generation.
"""

from pathlib import Path

from kissim.comparison import kissim_to_newick

def tree(inputfile, outputfile):
    """
    Generate kissim-based kinase tree. 

    Parameters
    ----------
    input_path : str or pathlib.Path
        Path to kissim kinase matrix (CSV file).
    output_path : str or pathlib.Path
        Path to kinase tree file (TREE file) in Newick format. 
    """

    #input_path = Path(input_path)
    #output_path = Path(output_path)

    kissim_to_newick(inputfile, outputfile)

