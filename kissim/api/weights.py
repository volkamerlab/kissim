"""
kissim.api.weights

Main API for kissim feature distances weighting to produce fingerprint distances.
"""

import logging
from pathlib import Path

import pandas as pd

from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator
from kissim.api.compare import weight_feature_distances

logger = logging.getLogger(__name__)


def weights(feature_distances_path, feature_weights=None, fingerprint_distances_path=None):
    """
    TODO
    """

    # Load feature distances
    feature_distances_path = Path(feature_distances_path)
    logger.info(f"Read feature distances from {feature_distances_path}...")
    feature_distances_generator = FeatureDistancesGenerator.from_csv(feature_distances_path)

    # Calculate fingerprint distances
    logger.info(f"Feature weights: {feature_weights}")
    fingerprint_distance_generator = FingerprintDistanceGenerator.from_feature_distances_generator(
        feature_distances_generator, feature_weights
    )

    # Optionally: Save to file
    if fingerprint_distances_path is not None:
        fingerprint_distances_path = Path(fingerprint_distances_path)
        logger.info(f"To file {fingerprint_distances_path}")
        fingerprint_distance_generator.to_csv(fingerprint_distances_path)

    return fingerprint_distance_generator
