"""
kissim.cli.encode

Encode structures (generate fingerprints) from CLI arguments.
"""

import numpy as np

from kissim.api import encode as api_encode
from kissim.cli.utils import configure_logger


def encode(args):
    """
    Encode structures.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    structure_klifs_ids = _parse_structure_klifs_ids(args.input)
    api_encode(structure_klifs_ids, args.output, args.ncores, args.local)


def _parse_structure_klifs_ids(args_input):
    """
    Parse structure KLIFS IDs.

    Parameters
    ----------
    args_input : list of str
        Either path to txt file with structure KLIFS ID (one ID per row) or one or more structure
        KLIFS IDs.

    Returns
    -------
    list of int
        List of structure KLIFS IDs.
    """

    if len(args_input) == 1:
        try:
            structure_klifs_ids = [int(args_input[0])]
        except ValueError:
            structure_klifs_ids = np.genfromtxt(fname=args_input[0], dtype=int).tolist()
    else:
        structure_klifs_ids = [int(i) for i in args_input]

    return structure_klifs_ids
