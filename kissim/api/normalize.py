"""
kissim.api.normalize

Main API for normalizing fingerprints.
"""

import logging
from pathlib import Path

from kissim.encoding import FingerprintGenerator, FingerprintGeneratorNormalized

logger = logging.getLogger(__name__)


def normalize(
    fingerprints_path, method="min_max", fine_grained=True, fingerprints_normalized_path=None
):
    """
    Remove outlier fingerprints (defined by spatial distances maximum).

    Parameters
    ----------
    fingerprints_path : str or pathlib.Path
        Path to fingerprints JSON file.
    method : str
        Normalization method.
    fine_grained : bool
        True (default):
            Distances: Calculate min/max per subpocket for each residue position individually.
            Moments: Calculate min/max per moment for each subpocket individually.
        False:
            Distances: Calculate min/max per subpocket over all residue positions.
            Moments: Calculate min/max per moment over all subpockets.
    fingerprints_normalized_path : str or pathlib.Path
        Path to normalized fingerprints JSON file.

    Returns
    -------
    kissim.encoding.FingerprintGenerator
        Normalized Fingerprints.
    """

    # Load fingerprints
    logger.info("Read fingerprints...")
    fingerprints_path = Path(fingerprints_path)
    fingerprint_generator = FingerprintGenerator.from_json(fingerprints_path)
    logger.info(f"Number of fingerprints: {len(fingerprint_generator.data)}")

    # Normalize fingerprints
    logger.info("Normalize fingerprints...")
    logger.info(f"Normalization method: {method}")
    logger.info(f"Use fine-grained normalization: {fine_grained}")
    fingerprint_generator_normalized = FingerprintGeneratorNormalized.from_fingerprint_generator(
        fingerprint_generator, method, fine_grained
    )
    if fingerprints_normalized_path is not None:
        fingerprints_normalized_path = Path(fingerprints_normalized_path)
        fingerprint_generator_normalized.to_json(fingerprints_normalized_path)
    logger.info(f"Number of fingerprints: {len(fingerprint_generator_normalized.data)}")

    return fingerprint_generator_normalized
