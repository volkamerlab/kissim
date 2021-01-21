"""
kissim.cli.encode

Compare encoded structures (fingerprints).
"""

from kissim.api import compare as api_compare
from kissim.cli.utils import configure_logger
from kissim.encoding import FingerprintGenerator


def compare(args):
    """
    Compare fingerprints.

    Parameters
    ----------
    args : argsparse.Namespace
        CLI arguments.
    """

    configure_logger(args.output)
    fingerprint_generator = FingerprintGenerator.from_json(args.input)
    weights = _parse_weights(args.weights)
    api_compare(
        fingerprint_generator,
        args.output,
        args.ncores,
        args.distance,
        weights,
    )


def _parse_weights(args_weights):
    """
    TODO
    """

    feature_weighting_schemes = {
        "100": [1.0, 0.0, 0.0],
        "010": [0.0, 1.0, 0.0],
        "001": [0.0, 0.0, 1.0],
        "110": [0.5, 0.5, 0.0],
        "101": [0.5, 0.0, 0.5],
        "011": [0.0, 0.5, 0.5],
        "111": [1.0 / 3, 1.0 / 3, 1.0 / 3],
    }

    weights = feature_weighting_schemes[args_weights]

    return weights
