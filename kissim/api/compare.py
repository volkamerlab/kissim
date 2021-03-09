"""
kissim.api.compare

Main API for kissim comparison.
"""

import logging

from kissim.comparison import FeatureDistancesGenerator, FingerprintDistanceGenerator

logger = logging.getLogger(__name__)


def compare(
    fingerprint_generator,
    csv_path=None,
    n_cores=1,
    distance_measure="scaled_euclidean",
    feature_weights="101",
):
    """
    Compare fingerprints (pairwise).

    Parameters
    ----------
    fingerprint_generator : kissim.encoding.FingerprintGenerator
        Fingerprints for KLIFS dataset.
    csv_path : str
        TODO
    n_cores : int
        Number of cores used to generate fingerprint distances.
    distance_measures : str
        Distance measures TODO.
    feature_weights : str
        Feature weighting scheme.
    """

    print(csv_path)

    start = datetime.datetime.now()

    logger.info(f"SIMILARITY: FingerprintDistanceGenerator: {feature_weights}")

    feature_distances_generator = FeatureDistancesGenerator.from_fingerprint_generator(
        fingerprint_generator
    )
    # TODO save to file

    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.from_feature_distances_generator(
        feature_distances_generator, feature_weights
    )
    # TODO save to file

    end = datetime.datetime.now()

    logger.info(f"Start of fingerprint distance generation: {start}")
    logger.info(f"End of fingerprint distance generation: {end}")
