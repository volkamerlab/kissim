"""
kissim.encoding.cli

CLI for fingerprint encoding
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from opencadd.databases.klifs import setup_remote, setup_local

from kissim.encoding import FingerprintGenerator


def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    TODO
        Command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        help="List of structure KLIFS IDs or path to txt file containing structure KLIFS IDs",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output json file containing fingerprint data",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--local",
        type=str,
        help="Path to KLIFS download folder. If set local KLIFS data is used, else remote KLIFS data",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--ncores",
        type=int,
        help="Number of cores. If 1 fingerprint generation in sequence, else in parallel.",
        required=False,
        default=1,
    )
    args = parser.parse_args()
    return args

def configure_logger(filename, level=logging.INFO):
    """TODO"""

    filename = Path(filename)
    logger = logging.getLogger("kissim")
    logger.setLevel(level)
    s_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename.parent / f"{filename.stem}.log")
    logger.addHandler(s_handler)
    logger.addHandler(f_handler)


def setup_klifs_session(args_local):
    """
    Set up KLIFS session. TODO
    """

    if args_local:
        klifs_session = setup_local(args_local)
    else:
        klifs_session = setup_remote()
    return klifs_session


def parse_structure_klifs_ids(args_input):
    """
    Parse structure KLIFS IDs. TODO
    """

    if len(args_input) == 1:
        try:
            structure_klifs_ids = [int(args_input[0])]
        except ValueError:
            structure_klifs_ids = np.genfromtxt(fname=args_input[0], dtype=int).tolist()
    else:
        structure_klifs_ids = [int(i) for i in args_input]

    return structure_klifs_ids


def main():

    args = parse_arguments()
    configure_logger(args.output)
    klifs_session = setup_klifs_session(args.local)
    structure_klifs_ids = parse_structure_klifs_ids(args.input)

    # Generate fingerprints
    fingerprints = FingerprintGenerator.from_structure_klifs_ids(
        structure_klifs_ids, klifs_session, args.ncores
    )

    # Save fingerprints to json file
    fingerprints.to_json(args.output)
