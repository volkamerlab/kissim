"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import logging

from . import FeatureDistancesGenerator, FingerprintDistanceGenerator

logger = logging.getLogger(__name__)

class Similarity:
    """
    Calculate all-against-all fingerprint distance.

    Attributes
    ----------
    path_results : pathlib.Path or str
        Path to results folder.
    """

    def __init__(self):

        self.path_results = None

    def execute(
        self, fingerprint_generator, distance_measures, feature_weighting_schemes, path_results
    ):
        """
        Calculate all-against-all feature and fingerprint distances for different distance measures
        and feature weighting schemes.

        Parameters
        ----------
        fingerprint_generator : encoding.FingerprintGenerator
            Fingerprints for KLIFS dataset.
        distance_measures : dict of str
            Distance measures: Key is name for file name, value is name as implemented in package.
        feature_weighting_schemes : dict of (dict or None)
            Feature weighting schemes: Key is name for file name, value is formatting as required
            for package.
        path_results : pathlib.Path or str
            Path to results folder.
        """

        # Set as class attributes as Path object
        self.path_results = Path(path_results)

        # Create results folder if not already there
        (self.path_results / "similarity").mkdir(parents=True, exist_ok=True)

        # All against all fingerprint comparison
        for distance_measure_name, distance_measure in distance_measures.items():

            # Generate feature distances (FeatureDistancesGenerator)
            feature_distances_generator = FeatureDistancesGenerator()
            feature_distances_generator.from_fingerprint_generator(fingerprint_generator)

            # Save class object to file
            with open(
                self.path_results / "similarity" / f"feature_distances_{distance_measure_name}.p",
                "wb",
            ) as f:
                pickle.dump(feature_distances_generator, f)

            for feature_weights_name, feature_weights in feature_weighting_schemes.items():
                # Generate fingerprint distance (FingerprintDistanceGenerator)
                fingerprint_distance_generator = FingerprintDistanceGenerator()
                fingerprint_distance_generator.from_feature_distances_generator(
                    feature_distances_generator, feature_weights
                )

                # Save class object to file
                with open(
                    self.path_results
                    / "similarity"
                    / f"fingerprint_distance_{distance_measure}_{feature_weights_name}.p",
                    "wb",
                ) as f:
                    pickle.dump(fingerprint_distance_generator, f)
