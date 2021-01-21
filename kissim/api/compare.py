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

    feature_distances_generator = FeatureDistancesGenerator()
    feature_distances_generator.from_fingerprint_generator(fingerprint_generator, distance_measure)
    # TODO save to file

    fingerprint_distance_generator = FingerprintDistanceGenerator()
    fingerprint_distance_generator.from_feature_distances_generator(
        feature_distances_generator, feature_weights
    )
    # TODO save to file
