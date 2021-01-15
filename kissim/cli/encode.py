"""
kissim.cli.encode

Encode structures (generate fingerprints) from CLI arguments.
"""

import numpy as np
from opencadd.databases.klifs import setup_remote, setup_local

from kissim.cli.utils import configure_logger
from kissim.encoding import FingerprintGenerator


def encode(args):
    """
    Encode structures.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    print(type(args))

    configure_logger(args.output)
    klifs_session = _setup_klifs_session(args.local)
    structure_klifs_ids = _parse_structure_klifs_ids(args.input)

    # Generate fingerprints
    fingerprints = FingerprintGenerator.from_structure_klifs_ids(
        structure_klifs_ids, klifs_session, args.ncores
    )

    # Save fingerprints to json file
    fingerprints.to_json(args.output)


def _setup_klifs_session(args_local):
    """
    Set up KLIFS session. 

    Parameters
    ----------
    args_local : str or None
        If path to local KLIFS download is given, set up local KLIFS session.
        If None is given, set up remote KLIFS session.

    Returns
    -------
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    """

    if args_local:
        klifs_session = setup_local(args_local)
    else:
        klifs_session = setup_remote()
    return klifs_session


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
