"""
kissim.comparison.feature_distances

Defines the feature distances for a fingerprint pair.
"""

import logging
from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial import distance

logger = logging.getLogger(__name__)

FEATURE_NAMES = {
    "physicochemical": [
        "size",
        "hbd",
        "hba",
        "charge",
        "aromatic",
        "aliphatic",
        "sco",
        "exposure",
    ],
    "distances": [
        "distance_to_centroid",
        "distance_to_hinge_region",
        "distance_to_dfg_region",
        "distance_to_front_pocket",
    ],
    "moments": ["moment1", "moment2", "moment3"],
}


class FeatureDistances:
    """
    Distances between two fingerprints for each of their features, plus details on feature type,
    feature, feature bit coverage, and feature bit number.

    Attributes
    ----------
    molecule_pair_code : tuple of str
        Codes of both molecules represented by the fingerprints.
    distances : np.ndarray
        Distances between two fingerprints for each of their features.
    bit_coverages : np.ndarray
        Bit coverages for two fingerprints for each of their features.
    """

    def __init__(self):

        self.molecule_pair_code = None
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
                chain.from_iterable([[key] * len(value) for key, value in FEATURE_NAMES.items()])
            )
            feature_names = list(chain.from_iterable(FEATURE_NAMES.values()))

            data_df = pd.DataFrame(
                {
                    "feature_type": feature_types,
                    "feature_name": feature_names,
                    "distance": self.distances,
                    "bit_coverage": self.bit_coverages,
                }
            )

            return data_df

    def from_fingerprints(self, fingerprint1, fingerprint2, distance_measure="scaled_euclidean"):
        """
        Calculate distance between two fingerprints for each (normalized) feature.

        Parameters
        ----------
        fingerprint1 : encoding.Fingerprint
            Fingerprint 1.
        fingerprint2 : encoding.Fingerprint
            Fingerprint 2.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.
        """

        # Set class attributes
        self.molecule_pair_code = (
            fingerprint1.structure_klifs_id,
            fingerprint2.structure_klifs_id,
        )

        # Get fingerprint (normalized or not normalized)
        f1 = fingerprint1.values_dict
        f2 = fingerprint2.values_dict

        # Iterate over all features and get feature type, feature name, feature distance and
        # feature bit coverage  # TODO this is only a quick fix, redo!
        distances = []
        bit_coverages = []
        f1, f2 = fingerprint1.physicochemical, fingerprint2.physicochemical
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = self.from_features(ff1, ff2, distance_measure)
            distances.append(distance)
            bit_coverages.append(bit_coverage)
        f1, f2 = fingerprint1.distances, fingerprint2.distances
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = self.from_features(ff1, ff2, distance_measure)
            distances.append(distance)
            bit_coverages.append(bit_coverage)
        f1, f2 = fingerprint1.moments.transpose(), fingerprint2.moments.transpose()
        for (_, ff1), (_, ff2) in zip(f1.items(), f2.items()):
            distance, bit_coverage = self.from_features(ff1, ff2, distance_measure)
            distances.append(distance)
            bit_coverages.append(bit_coverage)

        self.distances = np.array(distances)
        self.bit_coverages = np.array(bit_coverages)

    def from_features(self, feature1, feature2, distance_measure="scaled_euclidean"):
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
            return self._scaled_euclidean_distance(feature_pair[0], feature_pair[1])

        elif distance_measure == "scaled_cityblock":
            return self._scaled_cityblock_distance(feature_pair[0], feature_pair[1])

        else:
            distance_measures = "scaled_euclidean scaled_cityblock".split()
            raise ValueError(
                f'Distance measure unknown. Choose from: {", ".join(distance_measures)}'
            )

    @staticmethod
    def _scaled_euclidean_distance(values1, values2):
        """
        Calculate scaled Euclidean distance between two value lists of same length.

        Parameters
        ----------
        values1 : np.ndarray
            Value list (same length as values2).
        values2 : np.ndarray
            Value list (same length as values1).

        Returns
        -------
        float
            Scaled Euclidean distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f"Distance calculation failed: Values lists are not of same length.")
        elif len(values1) == 0:
            return np.nan
        else:
            d = 1 / len(values1) * distance.euclidean(values1, values2)
            return d

    @staticmethod
    def _scaled_cityblock_distance(values1, values2):
        """
        Calculate scaled cityblock distance between two value lists of same length.

        Parameters
        ----------
        values1 : np.ndarray
            Value list (same length as values2).
        values2 : np.ndarray
            Value list (same length as values1).

        Returns
        -------
        float
            Scaled cityblock distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f"Distance calculation failed: Values lists are not of same length.")
        elif len(values1) == 0:
            return np.nan
        else:
            d = 1 / len(values1) * distance.cityblock(values1, values2)
            return d
