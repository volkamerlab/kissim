"""
kissim.comparison.feature_distances

Defines the feature distances for a fingerprint pair.
"""

import logging
from itertools import chain

import numpy as np
import pandas as pd

from ..schema import DISTANCES_FEATURE_NAMES
from .utils import scaled_euclidean_distance, scaled_cityblock_distance

logger = logging.getLogger(__name__)


class FeatureDistances:
    """
    Distances between two fingerprints for each of their features, plus details on feature type,
    feature, feature bit coverage, and feature bit number.

    Attributes
    ----------
    structure_pair_ids : tuple of str or int
        IDs for structures that are representated by the input fingerprints.
    kinase_pair_ids : tuple of str or int
        IDs for kinases that are represented by the input fingerprints.
    distances : np.ndarray
        Distances between two fingerprints for each of their features.
    bit_coverages : np.ndarray
        Bit coverages for two fingerprints for each of their features.
    """

    def __init__(self):

        self.structure_pair_ids = None
        self.kinase_pair_ids = None
        self.distances = None
        self.bit_coverages = None

    @property
    def data(self):
        """
        Feature distances for fingerprint pair, with details on feature types, feature names, and
        feature bit coverages.

        Returns
        -------
        pandas.DataFrame
            Feature distances for fingerprint pair with details on feature types, features names,
            and feature bit coverages.
        """

        if (self.distances is not None) and (self.bit_coverages is not None):

            feature_types = list(
                chain.from_iterable(
                    [[key] * len(value) for key, value in DISTANCES_FEATURE_NAMES.items()]
                )
            )
            feature_names = list(chain.from_iterable(DISTANCES_FEATURE_NAMES.values()))

            data_df = pd.DataFrame(
                {
                    "feature_type": feature_types,
                    "feature_name": feature_names,
                    "distance": self.distances,
                    "bit_coverage": self.bit_coverages,
                }
            )

            return data_df

    @classmethod
    def from_fingerprints(cls, fingerprint1, fingerprint2):
        """
        Calculate distance between two fingerprints for each (normalized) feature.

        Parameters
        ----------
        fingerprint1 : encoding.Fingerprint
            Fingerprint 1.
        fingerprint2 : encoding.Fingerprint
            Fingerprint 2.

        Returns
        -------
        kissim.comparison.feature_distances
            Feature distances.
        """

        feature_distances = cls()

        # Set class attributes
        feature_distances.structure_pair_ids = (
            fingerprint1.structure_klifs_id,
            fingerprint2.structure_klifs_id,
        )
        feature_distances.kinase_pair_ids = (fingerprint1.kinase_name, fingerprint2.kinase_name)

        # Get fingerprint (normalized or not normalized)
        f1 = fingerprint1.values_dict
        f2 = fingerprint2.values_dict

        # Iterate over all features and get feature type, feature name, feature distance and
        # feature bit coverage  # TODO this is only a quick fix, redo!
        distances = []
        bit_coverages = []

        # Physicochemical features
        f1, f2 = fingerprint1.physicochemical, fingerprint2.physicochemical
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = feature_distances._get_feature_distances_and_bit_coverages(
                ff1, ff2, "scaled_cityblock"
            )
            distances.append(distance)
            bit_coverages.append(bit_coverage)

        # Distances features
        f1, f2 = fingerprint1.distances, fingerprint2.distances
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = feature_distances._get_feature_distances_and_bit_coverages(
                ff1, ff2, "scaled_euclidean"
            )
            distances.append(distance)
            bit_coverages.append(bit_coverage)

        # Moments features
        f1, f2 = fingerprint1.moments.transpose(), fingerprint2.moments.transpose()
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = feature_distances._get_feature_distances_and_bit_coverages(
                ff1, ff2, "scaled_euclidean"
            )
            distances.append(distance)
            bit_coverages.append(bit_coverage)

        feature_distances.distances = np.array(distances)
        feature_distances.bit_coverages = np.array(bit_coverages)

        return feature_distances

    @classmethod
    def _from_dict(cls, feature_distances_dict):
        """
        Initiate the feature distances from a dictionary containing the FeatureDistances class
        attributes.

        Parameters
        ----------
        feature_distances_dict : dict
            FeatureDistances attributes in the form of a dictionary.
        """

        feature_distances = cls()
        feature_distances.structure_pair_ids = tuple(feature_distances_dict["structure_pair_ids"])
        feature_distances.kinase_pair_ids = tuple(feature_distances_dict["kinase_pair_ids"])
        feature_distances.distances = np.array(feature_distances_dict["distances"], dtype=np.float)
        feature_distances.bit_coverages = np.array(
            feature_distances_dict["bit_coverages"], dtype=np.float
        )
        return feature_distances

    def _get_feature_distances_and_bit_coverages(
        self, feature1, feature2, distance_measure="scaled_euclidean"
    ):
        """
        Distance and bit coverage for a feature pair.

        Parameters
        ----------
        feature1 : pd.Series
            Feature bits for a given feature in fingerprint 1.
        feature2 : pd.Series
            Feature bits for a given feature in fingerprint 2.
        distance_measure : str
            Distance measure.

        Returns
        -------
        tuple of float
            Distance and bit coverage value for a feature pair.
        """

        if len(feature1) != len(feature2):
            raise ValueError(f"Features are not of same length!")

        # Cast feature pair to numpy array
        feature_pair = np.array([feature1, feature2])

        # Remove NaN positions in feature pair
        feature_pair_wo_nan = feature_pair[:, ~np.isnan(feature_pair).any(axis=0)]

        # Get feature pair coverage
        bit_coverage = round(feature_pair_wo_nan.shape[1] / feature_pair.shape[1], 2)

        # Get feature distance
        distance = self._calculate_feature_distance(feature_pair_wo_nan, distance_measure)

        return distance, bit_coverage

    def _calculate_feature_distance(self, feature_pair, distance_measure="scaled_euclidean"):
        """
        Calculate distance between two value lists (describing each the same feature).

        Parameters
        ----------
        feature_pair : np.ndarray
            Pairwise bits of one feature extracted from two fingerprints (only bit positions
            without any NaN value).
        distance_measure : str
            Distance measure.

        Returns
        -------
        float
            Distance between two value lists (describing each the same feature).
        """

        # Test if parameter input is correct
        if not isinstance(feature_pair, np.ndarray):
            raise TypeError(
                f'Parameter "feature_pair" must be of type np.ndarray, but is {type(feature_pair)}.'
            )

        # Set feature distance to NaN if no bits available for distance calculation
        if len(feature_pair) == 0:
            return np.nan

        if feature_pair.shape[0] != 2:
            raise ValueError(
                f'Parameter "feature_pair" has not two (i.e. {feature_pair.shape[1]}) '
                f"np.ndarray rows."
            )

        # Get feature distance
        if distance_measure == "scaled_euclidean":
            return scaled_euclidean_distance(feature_pair[0], feature_pair[1])

        elif distance_measure == "scaled_cityblock":
            return scaled_cityblock_distance(feature_pair[0], feature_pair[1])

        else:
            distance_measures = "scaled_euclidean scaled_cityblock".split()
            raise ValueError(
                f'Distance measure unknown. Choose from: {", ".join(distance_measures)}'
            )
