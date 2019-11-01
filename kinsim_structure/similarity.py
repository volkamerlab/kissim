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


def get_fingerprint_type1_similarity(pair, measure='ballester', weight=0.5):
    """
    Get similarity score for fingerprint type1 (consisting of physicochemical and distance properties) based on a
    similarity measure (default modified Manhattan distance).
    Option to weight physicochemical and distance properties differently (default None).

    Parameters
    ----------
    pair : 2-element-list of kinsim_structure.encoding.Fingerprint
        Fingerprint pair.
    measure : str
        Similarity measure name.
    weight : float
        Similarities for physicochemical and distance fingerprint bits are calculated separately, and
        summed up with respect to assigned weight for the physicochemical part and (1-weight) for the distance part.

    Returns
    -------
    List
        List of molecule names in pair and their score.
    """

    if 0 <= weight <= 1:

        score_physchem, coverage_physchem = calculate_similarity(
            pair[0].fingerprint_type1_normalized[FEATURE_NAMES[:8]],
            pair[1].fingerprint_type1_normalized[FEATURE_NAMES[:8]],
            measure=measure
        )
        score_distances, coverage_distances = calculate_similarity(
            pair[0].fingerprint_type1_normalized[FEATURE_NAMES[8:]],
            pair[1].fingerprint_type1_normalized[FEATURE_NAMES[8:]],
            measure=measure
        )

        score = weight * score_physchem + (1-weight) * score_distances

        return [
            pair[0].molecule_code,
            pair[1].molecule_code,
            score,
            score_physchem,
            score_distances,
            None,
            coverage_physchem,
            coverage_distances
        ]

    else:
        raise ValueError(f'Weight must be between 0 and 1. Given weight is: {weight}')


def get_fingerprint_type2_similarity(pair, measure='ballester', weight=0.5):

    if 0 <= weight <= 1:

        score_physchem, coverage_physchem = calculate_similarity(
            pair[0].fingerprint_type2_normalized['physchem'],
            pair[1].fingerprint_type2_normalized['physchem'],
            measure=measure
        )
        score_moments, coverage_moments = calculate_similarity(
            pair[0].fingerprint_type2_normalized['moments'],
            pair[1].fingerprint_type2_normalized['moments'],
            measure=measure
        )

        score = weight * score_physchem + (1 - weight) * score_moments

        return [
            pair[0].molecule_code,
            pair[1].molecule_code,
            score,
            score_physchem,
            score_moments,
            None,
            coverage_physchem,
            coverage_moments
        ]

    else:
        raise ValueError(f'Weight must be between 0 and 1. Given weight is: {weight}')


def calculate_similarity(fingerprint1, fingerprint2, measure='euklidean'):
    """
    Calculate the similarity between two fingerprints based on a similarity measure.

    Parameters
    ----------
    fingerprint1 : pandas.DataFrame
        Fingerprint for molecule.
    fingerprint2 : pandas.DataFrame
        Fingerprint for molecule.
    measure : str
        Similarity measurement method:
         - ballester (inverse of the translated and scaled Manhattan distance)
    Returns
    -------
    tuple of (float, int)
        Similarity score and coverage (ratio of bits used for similarity score).
    """

    measures = 'ballester manhattan euclidean'.split()

    # Convert DataFrame into 1D array
    if isinstance(fingerprint1, pd.DataFrame) and isinstance(fingerprint2, pd.DataFrame):
        fingerprint1 = fingerprint1.values.flatten()
        fingerprint2 = fingerprint2.values.flatten()
    else:
        raise ValueError(f'Input fingerprints must be of type pandas.DataFrame '
                         f'but are {type(fingerprint1)} (fp1) and {type(fingerprint2)} (fp2).')

    if len(fingerprint1) != len(fingerprint2):
        raise ValueError(f'Input fingerprints must be of same length.')
    else:
        pass

    # Merge both fingerprints to array in order to remove positions with nan values
    fingerprints = np.array(
        [
            fingerprint1,
            fingerprint2
        ]
    ).transpose()

    # Remove nan positions (shall not be compared)
    fingerprints_reduced = fingerprints[
        ~np.isnan(fingerprints).any(axis=1)
    ]

    # Get number of bits that can be compared (after nan bits removal)
    coverage = fingerprints_reduced.shape[0] / float(fingerprints.shape[0])

    if coverage == 0:
        score = None
        return score, coverage
    else:
        pass

    fp1 = fingerprints_reduced[:, 0]
    fp2 = fingerprints_reduced[:, 1]

    if measure == measures[0]:  # Inverse of the translated and scaled Manhattan distance
        score = 1 / (1 + 1 / len(fp1) * distance.cityblock(fp1, fp2))
        return score, coverage

    elif measure == measures[1]:  # Scaled Manhattan distance
        score = 1 - 1 / len(fp1) * distance.cityblock(fp1, fp2)
        return score, coverage

    elif measure == measures[2]:  # Euclidean distance
        score = 1 - 1 / len(fp1) * distance.euclidean(fp1, fp2)
        return score, coverage

    else:
        raise ValueError(f'Please choose a similarity measure: {", ".join(measures)}')


def _calc_fingerprint_distance(
    feature_distances,
    feature_name_physicochemical='physicochemical',
    feature_name_spatial='moments',
    feature_weights=None
):
    """

    Parameters
    ----------
    feature_distances : dict of list
        xxx
    feature_name_physicochemical : str
        xxx
    feature_name_spatial : str
        xxx
    feature_weights : dict of list
        xxx

    Returns
    -------
    float
        Fingerprint distance.
    """

    feature_weights_default = {
        'physicochemical': [1.0] * 8,
        'distances': [1.0] * 4,
        'moments': [1.0] * 3
    }

    if feature_weights:
        feature_weights = feature_weights_default
    else:
        if not list(feature_weights.keys()) == list(feature_weights_default.keys()):
            raise ValueError(f'Keys in feature_weights dict are wrong. '
                             f'Parameter needs to be in the following form: {feature_weights_default}')
        elif not [len(value) for key, value in feature_weights.items()] == [8, 4, 3]:
            raise ValueError(f'Lists in feature_weights are of wrong length. '
                             f'Parameter needs to be in the following form: {feature_weights_default}')
        else:
            pass

    # Select features
    selected_distances = feature_distances[feature_name_physicochemical] + feature_distances[feature_name_spatial]
    selected_distances = pd.Series(selected_distances)

    # Select feature weights
    selected_weights = feature_weights[feature_name_physicochemical] + feature_weights[feature_name_spatial]
    selected_weights = pd.Series(selected_weights)
    selected_weights = selected_weights / selected_weights.sum()  # Scale sum of all feature weights to 1

    return (selected_distances * selected_weights).sum()


class FeatureDistancesGenerator:
    """

    """

    def __init__(self):

        self.molecule_codes = None
        self.data = {
            'physicochemical': [],
            'distances': [],
            'moments': []
        }

    def from_fingerprint_pair(self, fingerprint1, fingerprint2, distance_measure='euclidean', normalized=True):
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
        normalized : bool
            Normalized (default) or non-normalized fingerprints.
        """

        self.molecule_codes = [fingerprint1.molecule_code, fingerprint2.molecule_code]

        if normalized:
            f1 = fingerprint1.fingerprint
            f2 = fingerprint2.fingerprint
        else:
            f1 = fingerprint1.fingerprint_normalized
            f2 = fingerprint2.fingerprint_normalized

        for feature_type in self.data.keys():

            distances_feature_category = []

            for feature_type in FEATURE_NAMES[feature_type]:

                distances_feature_category.append(
                    self._calc_feature_distance(
                        f1[feature_type][feature_type],
                        f2[feature_type][feature_type],
                        distance_measure
                    )
                )

            self.data[feature_type] = distances_feature_category

    def _calc_feature_distance(self, feature_values1, feature_values2, distance_measure='euclidean'):
        """
        Calculate distance between two value lists (describing each the same feature).

        Parameters
        ----------
        feature_values1 : list or pandas.Series
            Value list (same length as values2).
        feature_values2 : list or pandas.Series
            Value list (same length as values1).
        distance_measure : str
            Distance measure.

        Returns
        -------
        float
            Distance between two value lists (describing each the same feature).
        """

        if isinstance(feature_values1, list):
            feature_values1 = pd.Series(feature_values1)
        if isinstance(feature_values2, list):
            feature_values2 = pd.Series(feature_values2)

        distance_measures = 'euclidean'.split()

        # Get feature values without nan positions
        values_reduced = self._get_values_without_nan(feature_values1, feature_values2)

        # Get distance
        if distance_measure == 'euclidean':
            return self._euclidean_distance(
                values_reduced['values'][0],
                values_reduced['values'][1]
            )
        else:
            raise ValueError(f'Similarity measure unknown. Choose from: {", ".join(distance_measures)}')

    @staticmethod
    def _extract_feature(fingerprint, feature_type):
        """
        Extract a feature from a fingerprint.

        Parameters
        ----------
        fingerprint : dict of pandas.DataFrame
            Fingerprint, i.e. physicochemical, distance and moment features.
        feature_type : str
            Name of feature type.

        Returns
        -------
        pd.Series
            Feature bits for a given feature type.
        """

        if feature_type not in FEATURE_NAMES['physicochemical'] + FEATURE_NAMES['distances'] + FEATURE_NAMES['moments']:
            raise ValueError(f'Feature could not be extracted: {feature_type}')

        feature = None

        for feature_category, feature_types in FEATURE_NAMES.items():

            if feature_type in feature_types:
                feature = fingerprint[feature_category][feature_type]

        return feature

    @staticmethod
    def _get_values_without_nan(values1, values2):
        """
        Get two value lists with all positions removed where one list or both lists contain nan values.

        Parameters
        ----------
        values1 : list or pandas.Series
            Value list (same length as values2).
        values2 : list or pandas.Series
            Value list (same length as values1).

        Returns
        -------
        dict
            Values without nan positions and values coverage (coverage of non nan values).
        """

        if len(values1) != len(values2):
            raise ValueError(f'Nan removal failed: Values lists are not of same length.')

        # Merge both fingerprints to array in order to remove positions with nan values
        values = np.array([values1, values2])

        # Remove nan positions (shall not be compared)
        values_reduced = values[:, ~np.any(np.isnan(values), axis=0)]

        # Get number of bits that can be compared (after nan bits removal)
        coverage = values_reduced.shape[0] / float(values.shape[1])

        return {
            'values': values_reduced,
            'coverage': coverage
        }

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

        d = 1 / len(values1) * distance.euclidean(values1, values2)

        return d
