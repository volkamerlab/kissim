"""
kissim.comparison.fingerprint_distance

Defines the distance for a fingerprint pair.
"""

import logging

from kissim.comparison import weights

logger = logging.getLogger(__name__)


class FingerprintDistance:
    """
    Distance between two fingerprints using feature-wise weighting.

    Attributes
    ----------
    structure_pair_ids : tuple of str or int
        IDs of both structures that are represented by the input fingerprints.
    kinase_pair_ids : tuple of str or int
        IDs for kinases that are represented by the input fingerprints.
    distance : float
        Fingerprint distance (weighted per feature).
    bit_coverage : float
        Fingerprint coverage (weighted per feature).
    feature_weights : np.array
        Weights set per feature.
    """

    def __init__(self):

        self.structure_pair_ids = None
        self.kinase_pair_ids = None
        self.distance = None
        self.bit_coverage = None
        self.feature_weights = None

    @classmethod
    def from_feature_distances(cls, feature_distances, feature_weights=None):
        """
        Get fingerprint distance.

        Parameters
        ----------
        feature_distances : kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
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

        Returns
        -------
        kissim.comparison.FingerprintDistance
            Fingerprint distance.
        """

        fingerprint_distance = cls()

        # Set class attributes
        fingerprint_distance.structure_pair_ids = feature_distances.structure_pair_ids
        fingerprint_distance.kinase_pair_ids = feature_distances.kinase_pair_ids

        # Add weights
        fingerprint_distance.feature_weights = weights.format_weights(feature_weights)

        # Calculate weighted sum of feature distances and feature coverage
        fingerprint_distance.distance = sum(
            feature_distances.distances * fingerprint_distance.feature_weights
        )
        fingerprint_distance.bit_coverage = sum(
            feature_distances.bit_coverages * fingerprint_distance.feature_weights
        )

        return fingerprint_distance
