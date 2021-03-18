"""
kissim.cli.compare

Compare encoded structures (fingerprints).
"""

from pathlib import Path

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

    configure_logger(Path(args.output) / "distances.log")
    fingerprint_generator = FingerprintGenerator.from_json(args.input)
    compare(fingerprint_generator, args.output, args.weights, args.ncores)
