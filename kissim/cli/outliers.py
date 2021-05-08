"""
kissim.cli.outliers

CLI to remove outliers from fingerprints (defined by spatial distances maximum).
"""

import numpy as np

from kissim.api import outliers
from kissim.cli.utils import configure_logger


def outliers_from_cli(args):
    """
    Remove fingerprint outliers.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    outliers(args.input, args.distance_max, args.output)
