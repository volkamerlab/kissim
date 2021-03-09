"""
kissim.comparison.fingerprint_distance

Defines the distance for a fingerprint pair.
"""

import logging

import numpy as np

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
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances,
                and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

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
        feature_weights_formatted = fingerprint_distance._format_weights(feature_weights)

        # Calculate weighted sum of feature distances and feature coverage
        fingerprint_distance.distance = sum(
            feature_distances.distances * feature_weights_formatted
        )
        fingerprint_distance.bit_coverage = sum(
            feature_distances.bit_coverages * feature_weights_formatted
        )

        return fingerprint_distance

    def _format_weights(self, feature_weights=None):
        """
        Get feature weights based on input weights (each feature or feature type can be set
        individually).

        Parameters
        ----------
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances,
                and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        np.ndarray
            Feature weights.
        """

        # The parameter feature_weights can come in three difference formats as described in this
        # method's docstring.
        # For each of the three formats perform a certain action:

        if feature_weights is None:  # Defaults to equally distributed weights between all features

            feature_weights = self._format_weight_per_feature(feature_weights)

        elif isinstance(feature_weights, list):

            if len(feature_weights) == 3:  # Set weights per feature type
                feature_weights = self._format_weight_per_feature_type(feature_weights)

            elif len(feature_weights) == 15:  # Set weights per feature
                feature_weights = self._format_weight_per_feature(feature_weights)

            else:
                raise ValueError(
                    f"Weights must have length 3 or 15, but have length {len(feature_weights)}."
                )

        else:

            raise TypeError(
                f'Data type of "feature_weights" parameter must be list, '
                f"but is {type(feature_weights)}."
            )

        return feature_weights

    @staticmethod
    def _format_weight_per_feature_type(feature_type_weights=None):
        """
        Distribute feature type weights equally to features per feature type.

        Parameters
        ----------
        feature_type_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances,
                and moments.
                All floats must sum up to 1.0.

        Returns
        -------
        np.ndarray
            Feature weights.
        """

        # 1. Either set feature weights to default or check if non-default input is correct
        if feature_type_weights is None:
            feature_type_weights = [1.0 / 3] * 3

        else:

            # Check data type of feature weights
            if not isinstance(feature_type_weights, list):
                raise TypeError(
                    f'Data type of "feature_weights" parameter must be list, but is '
                    f"{type(feature_type_weights)}."
                )

            # Check if feature weight keys are correct
            if len(feature_type_weights) != 3:
                raise ValueError(
                    f"List must have length 3, but has length {len(feature_type_weights)}."
                )

            # Check if sum of weights is 1.0
            if sum(feature_type_weights) != 1.0:
                raise ValueError(
                    f"Sum of all weights must be one, but is {sum(feature_type_weights)}."
                )

        # 2. Distribute feature type weight equally to features in feature type
        # (in default feature order)
        feature_weights_formatted = []

        for feature_type_weight, n_features_per_type in zip(feature_type_weights, [8, 4, 3]):
            feature_weights_formatted.extend(
                [feature_type_weight / n_features_per_type] * n_features_per_type
            )

        return np.array(feature_weights_formatted)

    @staticmethod
    def _format_weight_per_feature(feature_weights=None):
        """
        Format feature weights.

        Parameters
        ----------
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
                All floats must sum up to 1.0.

        Returns
        -------
        np.ndarray
            Feature weights.
        """

        # 1. Either set feature weights to default or check if non-default input is correct
        if feature_weights is None:
            feature_weights = [1.0 / 15] * 15

        else:

            # Check data type of feature weights
            if not isinstance(feature_weights, list):
                raise TypeError(
                    f'Data type of "feature_weights" parameter must be list, but is '
                    f"{type(feature_weights)}."
                )

            # Check if feature weight keys are correct
            if len(feature_weights) != 15:
                raise ValueError(
                    f"List must have length 15, but has length {len(feature_weights)}."
                )

            # Check if sum of weights is 1.0
            if sum(feature_weights) != 1.0:
                raise ValueError(f"Sum of all weights must be one, but is {sum(feature_weights)}.")

        return np.array(feature_weights)
