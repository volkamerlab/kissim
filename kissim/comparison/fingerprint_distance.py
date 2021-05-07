"""
kissim.comparison.fingerprint_distance

Defines the distance for a fingerprint pair.
"""

import logging

import numpy as np

from kissim.comparison.utils import format_weights

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
    """

    def __init__(self):

        self.structure_pair_ids = None
        self.kinase_pair_ids = None
        self.distance = None
        self.bit_coverage = None

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

        # Get data of interest from input
        weights = format_weights(feature_weights)
        bit_coverages = feature_distances.bit_coverages
        distances = feature_distances.distances

        # Set class attributes
        fingerprint_distance.structure_pair_ids = feature_distances.structure_pair_ids
        fingerprint_distance.kinase_pair_ids = feature_distances.kinase_pair_ids

        # Calculate weighted sum of feature bit coverages
        fingerprint_distance.distance = fingerprint_distance._distance(distances, weights)
        # Calculate weighted sum of feature distances
        fingerprint_distance.bit_coverage = fingerprint_distance._bit_coverage(
            bit_coverages, weights
        )

        return fingerprint_distance

    def _distance(self, distances, weights):
        """
        Weighte sum of distances (weights recalibrated in case distances contain NaN values).

        Parameters
        ----------
        distances : np.ndarray
            Distances vector. Same length as weights vector.
        weights : np.ndarray
            Weights vector. Same length as values vector.

        Returns
        -------
        float
            Weighted sum of distances.
        """

        if np.isnan(distances).any():
            (
                distances,
                weights,
            ) = self._remove_nan_distances_and_recalibrate_weights(distances, weights)
        distance = self._calculate_weighted_sum(distances, weights)
        return distance

    def _bit_coverage(self, bit_coverages, weights):
        """
        Weighte sum of bit coverages.

        Parameters
        ----------
        bit_coverages : np.ndarray
            Bit coverages vector. Same length as weights vector.
        weights : np.ndarray
            Weights vector. Same length as values vector.

        Returns
        -------
        float
            Weighted sum of bit coverages.
        """

        bit_coverage = self._calculate_weighted_sum(bit_coverages, weights)
        return bit_coverage

    @staticmethod
    def _calculate_weighted_sum(values, weights):
        """
        Calculate the weighted sum of values.

        Parameters
        ----------
        values : np.ndarray
            Values vector. Same length as weights vector.
        weights : np.ndarray
            Weights vector. Same length as values vector.

        Returns
        -------
        float
            Weighted sum of values.
        """

        if np.isnan(values).any():
            raise ValueError(f"Input values cannot contain NaN values: {values}")

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError(f"Sum of input weights must be 1 but is {np.sum(weights)}.")

        return np.sum(values * weights)

    @staticmethod
    def _remove_nan_distances_and_recalibrate_weights(distances, weights):
        """
        Remove NaN values from distances and recalibrate weights.

        Parameters
        ----------
        distances : np.ndarray
            Distances vector. Same length as weights vector.
        weights : np.ndarray
            Weights vector. Same length as distances vector.

        Returns
        -------
        distances_wo_nan : np.ndarray
            Distances vector without NaN values.
        weights_wo_nan_recalibrated : np.ndarray
            Weights vector without weights for NaN distances values, while recalibrating the
            remaining weights.

        Notes
        -----

        Weights recalibration: Weights for NaN distance values are distributed across
        not-NaN distance values - distribution w.r.t. to the corresponding weights for not-NaN
        distance values.

        Example:

        distances = np.array([np.nan, 0.1, 0.2, 0.8, 0.9])
        weights = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
        >>>
        distances_wo_nan = np.array([0.1, 0.2, 0.8, 0.9])
        weights_wo_nan_recalibrated = np.array([0.2, 0.3, 0.3, 0.1]) +
                                      np.array([0.2, 0.3, 0.3, 0.1]) * 0.1 / 0.9

        where
        - 0.1 is the sum of weights for NaN distances
        - 0.9 is the sum of weights for not-Nan distances
        - np.array([0.2, 0.3, 0.3, 0.1]) * 0.1 / 0.9 == 0.1
        """

        # Remove NaN values from distances
        distances_wo_nan = distances[~np.isnan(distances)]

        # Split weights by not-NaN and NaN distances
        weights_wo_nan = weights[~np.isnan(distances)]
        weights_w_nan = weights[np.isnan(distances)]

        # Recalibrate weights (distribute weights for NaN distances across not-NaN distances)
        weights_wo_nan_recalibrated = weights_wo_nan + weights_wo_nan * (
            np.sum(weights_w_nan) / np.sum(weights_wo_nan)
        )

        return distances_wo_nan, weights_wo_nan_recalibrated
