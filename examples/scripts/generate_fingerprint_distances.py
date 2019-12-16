"""
generate_fingerprint_distances.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Generate the fingerprint distances for all against all fingerprints.
"""

import logging
from pathlib import Path
import pickle
import sys

sys.path.append('../..')
from kissim.similarity import FeatureDistancesGenerator, FingerprintDistanceGenerator

PATH_TO_KINSIM = Path('.') / '..' / '..'
PATH_TO_FINGERPRINTS = PATH_TO_KINSIM / 'examples' / 'results' / 'fingerprints' / 'fingerprints.p'
PATH_TO_RESULTS = PATH_TO_KINSIM / 'examples' / 'results' / 'similarity'

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    filename=PATH_TO_RESULTS / 'generate_fingerprint_distances.log',
    filemode='w',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def main():

    # Load fingerprints (FingerprintGenerator)
    with open(PATH_TO_FINGERPRINTS, 'rb') as f:
        fingerprint_generator = pickle.load(f)

    # Set parameters
    distance_measures = {
        'scaledEuclidean': 'scaled_euclidean',
        'scaledCityblock': 'scaled_cityblock'
    }
    feature_weighting_schemes = {
        'weights100': {'physicochemical': 1.0, 'distances': 0.0, 'moments': 0.0},
        'weights010': {'physicochemical': 0.0, 'distances': 1.0, 'moments': 0.0},
        'weights001': {'physicochemical': 0.0, 'distances': 0.0, 'moments': 1.0},
        'weights110': {'physicochemical': 0.5, 'distances': 0.5, 'moments': 0.0},
        'weights101': {'physicochemical': 0.5, 'distances': 0.0, 'moments': 0.5},
        'weights011': {'physicochemical': 0.0, 'distances': 0.5, 'moments': 0.5},
        'weights111': {'physicochemical': 1.0 / 3, 'distances': 1.0 / 3, 'moments': 1.0 / 3}
    }

    # All against all fingerprint comparison
    for distance_measure_name, distance_measure in distance_measures.items():

        # Generate feature distances (FeatureDistancesGenerator)
        logger.info(f'***FeatureDistancesGenerator: {distance_measure_name}')

        feature_distances_generator = FeatureDistancesGenerator()
        feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

        with open(PATH_TO_RESULTS / f'feature_distances_{distance_measure_name}.p', 'wb') as f:
            pickle.dump(feature_distances_generator, f)

        for feature_weights_name, feature_weights in feature_weighting_schemes.items():

            # Generate fingerprint distance (FingerprintDistanceGenerator)
            logger.info(f'***FingerprintDistanceGenerator: {distance_measure_name}')

            fingerprint_distance_generator = FingerprintDistanceGenerator()
            fingerprint_distance_generator.from_feature_distances_generator(
                feature_distances_generator,
                feature_weights
            )

            with open(PATH_TO_RESULTS / f'fingerprint_distance_{distance_measure}_{feature_weights_name}.p', 'wb') as f:
                pickle.dump(feature_distances_generator, f)


if __name__ == "__main__":

    main()
