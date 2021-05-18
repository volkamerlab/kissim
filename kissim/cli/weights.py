"""
kissim.cli.weights

CLI to perform kissim feature distances weighting to produce fingerprint distances.
"""

from kissim.api import weights
from kissim.cli.utils import configure_logger


def weights_from_cli(args):
    """
    Weight feature distances to generate fingerprint distances.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    weights(args.input, args.weights, args.output)
