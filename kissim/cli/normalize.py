"""
kissim.cli.normalize

Normalize fingerprints from CLI arguments.
"""

from kissim.api import normalize
from kissim.cli.utils import configure_logger


def normalize_from_cli(args):
    """
    Normalize fingerprints.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    normalize(args.input, args.method, bool(args.fine_grained), args.output)
