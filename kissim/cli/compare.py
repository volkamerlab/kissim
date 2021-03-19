"""
kissim.cli.encode

Compare encoded structures (fingerprints).
"""

from kissim.api import compare
from kissim.cli.utils import configure_logger
from kissim.encoding import FingerprintGenerator


def compare_from_cli(args):
    """
    Compare fingerprints.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    fingerprint_generator = FingerprintGenerator.from_json(args.input)
    compare(fingerprint_generator, args.output, args.ncores, args.weights)
