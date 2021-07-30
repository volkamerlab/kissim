"""
kissim.cli.subset

CLI to subset fingerprints.
"""

from kissim.api import subset
from kissim.cli.utils import configure_logger


def subset_from_cli(args):
    """
    Remove fingerprint outliers.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    subset(args.input, args.subset, args.output)
