"""
kissim.cli.main

Main command line interface (CLI) script defining sub-commands and arguments.

Resources:
- https://gist.github.com/lusis/624782
- https://docs.python.org/3/library/argparse.html#sub-commands

Special thanks to @jaimergp for the pointers.
"""

import argparse
import subprocess

from kissim.cli import encode_from_cli, compare_from_cli, tree_from_cli


def main():
    """
    Define CLI sub-commands and their arguments.
    Note: Package entry point points to this function.

    Sub-commands are:
    - encode
    - compare
    - tree
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    encode_subparser = subparsers.add_parser("encode")
    compare_subparser = subparsers.add_parser("compare")
    tree_subparser = subparsers.add_parser("tree")

    # Arguments and function to be called for sub-command encode
    encode_subparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        help="List of structure KLIFS IDs or path to txt file containing structure KLIFS IDs.",
        required=True,
    )
    encode_subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output json file containing fingerprint data.",
        required=True,
    )
    encode_subparser.add_argument(
        "-l",
        "--local",
        type=str,
        help="Path to KLIFS download folder. If set local KLIFS data is used, else remote KLIFS data.",
        required=False,
    )
    encode_subparser.add_argument(
        "-c",
        "--ncores",
        type=int,
        help="Number of cores. If 1 fingerprint generation in sequence, else in parallel.",
        required=False,
        default=1,
    )
    encode_subparser.set_defaults(func=encode_from_cli)

    # Arguments and function to be called for sub-command compare
    compare_subparser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to json file containing fingerprint data.",
        required=True,
    )
    compare_subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output folder where distance json files will be saved.",
        required=True,
    )
    compare_subparser.add_argument(
        "-w",
        "--weights",
        type=float,
        nargs=15,
        help="Feature weights. Eeach feature must be set individually, all weights must sum up to 1.0.",
        required=False,
    )
    compare_subparser.add_argument(
        "-c",
        "--ncores",
        type=int,
        help="Number of cores. If 1 comparison in sequence, else in parallel.",
        required=False,
        default=1,
    )
    compare_subparser.set_defaults(func=compare_from_cli)


    # Arguments and function to be called for sub-command encode
    tree_subparser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to a kissim kinase matrix CSV file.",
        required=True,
    )
    tree_subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output file in Newick format.",
        required=True,
    )
    tree_subparser.set_defaults(func=tree_from_cli)

    args = parser.parse_args()
    try: 
        args.func(args)
    except AttributeError:
        # Run help if no arguments were given
        subprocess.run(["kissim", "-h"])
