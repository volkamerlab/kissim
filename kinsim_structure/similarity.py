"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial import distance

from kinsim_structure.encoding import FEATURE_NAMES

logger = logging.getLogger(__name__)

FEATURE_DISTANCES_FORMAT = {
    'physicochemical': [None]*8,
    'distances': [None]*4,
    'moments': [None]*3
}


class FingerprintDistance:

    def __init__(self):
        self.molecule_codes = None
        self.data = None

    def from_feature_distances(self, feature_distances, feature_weights=None):
        """
        Get fingerprint distance.

        Parameters
        ----------
        feature_distances : similarity.FeatureDistances
            Feature distances.
        feature_weights : (dict of float or int) or (dict of list of float or int) or None
            (a) Dictionary of weight (value) per feature type (key), i.e. physicochemical, distances, moments).
            (b) Dictionary of weights for each feature: Dict keys are feature types (physicochemical, distances, moments)
            and dict values are list of float/int describing weights for each feature
            (physicochemical: 8 features, distances: 4 features, moments: 3 values).
            All floats must sum up to 1.0.

        Returns
        -------
        float
            Fingerprint distance.
        """

        self.molecule_codes = feature_distances.molecule_codes
        feature_distances = feature_distances.data

        feature_distances = self._add_weight_column(feature_distances, feature_weights)

        fingerprint_distance = (feature_distances.distance * feature_distances.weights).sum()

        self.data = fingerprint_distance

    def _add_weight_column(self, feature_distances, feature_weights=None):
        """
        Add feature weights to feature distance details (each feature type OR feature can be set individually).

        Parameters
        ----------
        feature_distances : pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : (dict of float or int) or (dict of list of float or int) or None
            (a) Dictionary of weight (value) per feature type (key), i.e. physicochemical, distances, moments).
            (b) Dictionary of weights for each feature: Dict keys are feature types (physicochemical, distances, moments)
            and dict values are list of float/int describing weights for each feature
            (physicochemical: 8 features, distances: 4 features, moments: 3 values).
            All floats must sum up to 1.0.

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, feature bit number, AND feature weights.

        """

        if feature_weights is None:
            return self._add_weight_per_feature_type(feature_distances, feature_weights)

        elif all([isinstance(i, (float, int)) for i in feature_weights.values()]):
            return self._add_weight_per_feature_type(feature_distances, feature_weights)

        elif all([isinstance(i, list) for i in feature_weights.values()]):
            return self._add_weight_per_feature(feature_distances, feature_weights)

        else:
            raise ValueError(f'Unknown input for which no exception is implemented. '
                             f'Please check docstring for details on input formats.')

    @staticmethod
    def _add_weight_per_feature_type(feature_distances, feature_weights=None):
        """
        Add feature weights to feature distance details (each feature type can be set individually).

        Parameters
        ----------
        feature_distances : pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : (dict of float or int) or None
            Dictionary of weight (value) per feature type (key), i.e. physicochemical, distances, moments).
            All floats must sum up to 1.0.

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, feature bit number, AND feature weights.
        """

        if feature_weights is None:

            feature_weights = {
                'physicochemical': 0.5,
                'distances': 0.5,
                'moments': 0.0
            }

        else:

            # Check if input weights have correct form
            weights_sum = sum(feature_weights.values())

            if weights_sum != 1.0:
                raise ValueError(f'Sum of all weights must be one, but is {weights_sum}.')

        # Cast weight values to float
        for feature_type, weight in feature_weights.items():

            if isinstance(feature_weights[feature_type], int):
                feature_weights[feature_type] = float(feature_weights[feature_type])

        # Set weights per feature
        weights = []

        for feature_type, group in feature_distances.groupby(by='feature_type', sort=False):
            weights.extend([feature_weights[feature_type] / len(group.distance)] * len(group.distance))

        # Add weights to DataFrame
        feature_distances['weights'] = weights

        return feature_distances

    @staticmethod
    def _add_weight_per_feature(feature_distances, feature_weights=None):
        """
        Add feature weights to feature distance details (each feature can be set individually).

        Parameters
        ----------
        feature_distances : pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : (dict of list of float or int) or None
            Dictionary of weights for each feature: Dict keys are feature types (physicochemical, distances, moments)
            and dict values are list of float/int describing weights for each feature
            (physicochemical: 8 features, distances: 4 features, moments: 3 values).
            All floats must sum up to 1.0.

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, feature bit number, AND feature weights.
        """

        if feature_weights is None:

            feature_weights = {
                'physicochemical': [0.0625] * 8,
                'distances': [0.125] * 4,
                'moments': [0.0] * 3
            }

        else:

            # Check if input weights are set to known feature types with needed number of weights
            if not all([feature_type in FEATURE_DISTANCES_FORMAT.keys() for feature_type in feature_weights.keys()]):
                raise ValueError(f'Feature weights (dict) have unknown feature types (keys). '
                                 f'Set the following feature types: '
                                 f'{", ".join(list(FEATURE_DISTANCES_FORMAT.keys()))}.')

            for feature_type, weights in feature_weights.items():
                if len(weights) != len(FEATURE_DISTANCES_FORMAT[feature_type]):
                    raise ValueError(f'Feature type "{feature_type}" has {len(weights)} weights, '
                                     f'but needs {len(FEATURE_DISTANCES_FORMAT[feature_type])} weights.')

            # Check if input weights have correct form
            weights_sum = sum([sum(i) for i in feature_weights.values()])

            if weights_sum != 1.0:
                raise ValueError(f'Sum of all weights must be one, but is {weights_sum}.')

        # Get weights as list and cast values to float
        weights = []

        for feature_type, weight in feature_weights.items():
            weights.extend([float(i) for i in feature_weights[feature_type]])

        # Add weights to DataFrame
        feature_distances['weights'] = weights

        return feature_distances


class FeatureDistances:
    """
    Distances between two fingerprints for each of their features, including information on feature bit coverage and
    bit number, in order to assess how reliable the feature distance is.

    Attributes
    ----------
    molecule_codes : list of str
        Codes of both molecules represented by the fingerprints.
    data : pandas.DataFrame
        Distances between two fingerprints for each of their features, plus details on feature type, feature,
        feature bit coverage, and feature bit number.
    """

    def __init__(self):

        self.molecule_codes = None
        self.data = None

    def from_fingerprints(self, fingerprint1, fingerprint2, distance_measure='euclidean'):
        """
        Calculate distance between two fingerprints for each feature.

        Parameters
        ----------
        fingerprint1 : encoding.Fingerprint
            Fingerprint 1.
        fingerprint2 : encoding.Fingerprint
            Fingerprint 2.
        distance_measure : str
            Type of distance measure.

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        """

        self.molecule_codes = [fingerprint1.molecule_code, fingerprint2.molecule_code]

        # Get fingerprint pair (normalized fingerprints only)
        fingerprint_pair = self._extract_fingerprint_pair(fingerprint1, fingerprint2, normalized=True)

        distances = []

        for feature_type in FEATURE_NAMES.keys():

            for feature_name in FEATURE_NAMES[feature_type]:

                # Get feature distance
                distance = self._calc_feature_distance(
                        fingerprint_pair[feature_type][feature_name],
                        distance_measure
                    )

                # Get number of feature bits without any NaN value
                bit_number = len(fingerprint_pair[feature_type][feature_name])

                # Get bit coverage
                bit_coverage = self._get_bit_coverage(feature_type, bit_number)

                # Save feature data to fingerprint data
                distances.append([feature_type, feature_name, distance, bit_coverage, bit_number])

        self.data = pd.DataFrame(
            distances,
            columns='feature_type feature_name distance bit_coverage bit_number'.split()
        )

    @staticmethod
    def _get_bit_coverage(feature_type, bit_number):
        """
        Get bit coverage for a given feature type.

        Parameters
        ----------
        feature_type : str
            Feature type: physicochemical, distances or moments.
        bit_number : int
            Number of feature bits used for distance calculation.

        Returns
        -------
        float
            Bit coverage describing the percentage of bits used for distance calculation.
        """

        if feature_type not in FEATURE_NAMES.keys():
            raise ValueError(f'Feature type unknown. Choose from: {", ".join(list(FEATURE_NAMES.keys()))}.')

        bit_number_moments = 4.0
        bit_number_other = 85.0

        if feature_type is 'moments':

            if 0 <= bit_number <= bit_number_moments:
                return round(bit_number / bit_number_moments, 2)
            else:
                raise ValueError(f'Unexcepted number of bits for {feature_type}: '
                                 f'Is {bit_number}, but must be between 0 and {int(bit_number_moments)}.')

        else:

            if 0 <= bit_number <= bit_number_other:
                return round(bit_number / bit_number_other, 2)
            else:
                raise ValueError(f'Unexcepted number of bits for {feature_type}: '
                                 f'Is {bit_number}, but must be between 0 and {int(bit_number_other)}.')

    def _calc_feature_distance(self, feature_pair, distance_measure='euclidean'):
        """
        Calculate distance between two value lists (describing each the same feature).

        Parameters
        ----------
        feature_pair : pandas.DataFrame
            Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
        distance_measure : str
            Distance measure.

        Returns
        -------
        dict
            Distance between two value lists (describing each the same feature).
        """

        distance_measures = 'euclidean'.split()

        if not isinstance(distance_measure, str):
            raise TypeError(f'Parameter "distance_measure" must be of type str, but is {type(distance_measure)}.')

        if distance_measure not in distance_measures:
            raise ValueError(f'Distance measure unknown. Choose from: {", ".join(distance_measures)}')

        if not isinstance(feature_pair, pd.DataFrame):
            raise TypeError(f'Parameter "feature_pair" must be of type pandas.DataFrame, but is {type(feature_pair)}.')

        if feature_pair.shape[1] != 2:
            raise ValueError(f'Parameter "feature_pair" must be pandas.DataFrame with two columns, '
                             f'but has {feature_pair.shape[1]} columns.')

        # Set feature distance to NaN if no bits available for distance calculation
        if len(feature_pair) == 0:
            return np.nan

        # In case there are still NaN positions, remove bit positions containing any NaN value
        feature_pair.dropna(how='any', axis=0, inplace=True)

        # Get feature distance
        if distance_measure == 'euclidean':
            return self._euclidean_distance(
                feature_pair.iloc[:, 0],  # Fingerprint 1
                feature_pair.iloc[:, 1]  # Fingerprint 2
            )

        else:
            raise ValueError(f'Distance measure unknown. Choose from: {", ".join(distance_measures)}')

    @staticmethod
    def _extract_fingerprint_pair(fingerprint1, fingerprint2, normalized=True):
        """
        For each feature, get both fingerprint bits without NaN positions.

        Parameters
        ----------
        fingerprint1 : encoding.Fingerprint
            Fingerprint 1.
        fingerprint2 : encoding.Fingerprint
            Fingerprint 2.
        normalized : bool
            Normalized (default) or non-normalized fingerprints.

        Returns
        -------
        dict of dict of pandas.DataFrame
            For each feature type, i.e. physicochemical, distances, and moments (dict) and for each corresponding
            feature, i.e. size, HBD, HDA, ... for physicochemical feature type (dict), non-NaN bits from both
            fingerprints (pandas.DataFrame).
        """

        if normalized:
            f1 = fingerprint1.fingerprint_normalized
            f2 = fingerprint2.fingerprint_normalized
        else:
            f1 = fingerprint1.fingerprint
            f2 = fingerprint2.fingerprint

        fingerprint_pair = {}

        # Iterate over all feature types
        for feature_type in FEATURE_NAMES.keys():

            fingerprint_pair[feature_type] = {}

            # Iterate over all features
            for feature_name in FEATURE_NAMES[feature_type]:

                # Concatenate feature bits from both fingerprints and remove bits where one or both bits are NaN
                feature_pair = pd.concat(
                    [f1[feature_type][feature_name], f2[feature_type][feature_name]],
                    axis=1
                )
                feature_pair.columns = ['fingerprint1', 'fingerprint2']
                feature_pair.dropna(how='any', axis=0, inplace=True)

                fingerprint_pair[feature_type][feature_name] = feature_pair

        return fingerprint_pair

    @staticmethod
    def _euclidean_distance(values1, values2):
        """
        Calculate Euclidean distance between two value lists of same length.

        Parameters
        ----------
        values1 : list or pandas.Series
            Value list (same length as values2).
        values2 : list or pandas.Series
            Value list (same length as values1).

        Returns
        -------
        float
            Euclidean distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f'Distance calculation failed: Values lists are not of same length.')
        elif len(values1) == 0:
            return np.nan
        else:
            d = 1 / len(values1) * distance.euclidean(values1, values2)
            return d

    @staticmethod
    def _cityblock_distance(values1, values2):
        """
        Calculate cityblock distance between two value lists of same length.

        Parameters
        ----------
        values1 : list or pandas.Series
            Value list (same length as values2).
        values2 : list or pandas.Series
            Value list (same length as values1).

        Returns
        -------
        float
            Euclidean distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f'Distance calculation failed: Values lists are not of same length.')
        elif len(values1) == 0:
            return np.nan
        else:
            d = 1 / len(values1) * distance.cityblock(values1, values2)
            return d
