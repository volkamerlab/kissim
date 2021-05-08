"""
kissim.api.weights

Main API for kissim feature distances weighting to produce fingerprint distances.
"""

import logging
from pathlib import Path

from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator

logger = logging.getLogger(__name__)


def weights(feature_distances_path, feature_weights=None, fingerprint_distances_path=None):
    """
    Apply feature distances weighting to calculate fingerprint distances.

    Parameters
    ----------
    feature_distances_path : str or pathlib.Path
        Path to feature distances CSV file.
    feature_weights : None or list of float
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15
            (15 features in total).
        (ii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic,
            aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
            distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            All floats must sum up to 1.0.
    fingerprint_distances_path : None or str or pathlib.Path
        Path to output fingerprint distances CSV file.

    Returns
    -------
    kissim.comparison.FingerprintDistanceGenerator
        Pairwise fingerprint distances.
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
