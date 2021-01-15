"""
kissim.cli.main

Main command line interface (CLI) script defining sub-commands (encode, compare) and arguments.

Resources:
- https://gist.github.com/lusis/624782
- https://docs.python.org/3/library/argparse.html#sub-commands

Special thanks to @jaimergp for the pointers.
"""

import argparse

from kissim.cli.encode import encode


def main():
    """
    Define CLI sub-commands and their arguments. 
    Note: Package entry point points to this function.

    Sub-commands are:
    - encode
    - compare
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    encode_subparser = subparsers.add_parser("encode")
    compare_subparser = subparsers.add_parser("compare")

    # Arguments and function to be called for sub-command encode
    encode_subparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        help="List of structure KLIFS IDs or path to txt file containing structure KLIFS IDs",
        required=True,
    )
    encode_subparser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to output json file containing fingerprint data",
        required=True,
    )
    encode_subparser.add_argument(
        "-l",
        "--local",
        type=str,
        help="Path to KLIFS download folder. If set local KLIFS data is used, else remote KLIFS data",
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
    encode_subparser.set_defaults(func=encode)

    # Arguments and function to be called for sub-command compare
    # TODO

    args = parser.parse_args()
    args.func(args)
