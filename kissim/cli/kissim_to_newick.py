"""kissim_to_newick

This is a small tool for processing a KISSIM similarity matrix into a clustered
kissim-based tree with assignment of the mean similarity to each branch.
The resulting tree is written to an output file in Newick format.
"""

import sys

from kissim.comparison import kissim_to_newick

def main(argv):
    """Main function for the kissim_to_newick tool as CLI."""

    if len(argv) != 2:
        print("Syntax: kissim_to_newick.py <inputfile> <outputfile>")
    else:
        inputfile = argv[0]
        outputfile = argv[1]
        kissim_to_newick(inputfile, outputfile)

if __name__ == "__main__":
    main(sys.argv[1:])