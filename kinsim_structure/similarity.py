"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool

from itertools import combinations, repeat
import numpy as np
import pandas as pd
from scipy.spatial import distance

from kinsim_structure.encoding import FEATURE_NAMES

logger = logging.getLogger(__name__)


class FingerprintDistanceGenerator:
    """
    Generate fingerprint distances for multiple fingerprint pairs based on their feature distances,
    given a feature weighting scheme.
    Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    distance_measure : str
        Type of distance measure, defaults to scaled Euclidean distance.
    molecule_codes : list of str
        Unique molecule codes associated with all fingerprints (sorted alphabetically).
    kinase_names : list of str
        Unique kinase names associated with all fingerprints (sorted alphabetically).
    feature_weights : dict of float or None
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15 (15 features in total).
        (ii) By feature type
            Feature types to be set are: physicochemical, distances, and moments.
        (iii) By feature:
            Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
            distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
            moment1, moment2, and moment3.
        For (ii) and (iii): All floats must sum up to 1.0.
    data : pandas.DataFrame
        Fingerprint distance and coverage, plus details on both molecule codes associated with fingerprint pairs.
    """

    def __init__(self):

        self.distance_measure = None
        self.feature_weights = None
        self.molecule_codes = None
        self.kinase_names = None
        self.data = None

    def from_feature_distances_generator(self, feature_distances_generator, feature_weights=None):
        """
        Generate fingerprint distances for multiple fingerprint pairs based on their feature distances,
        given a feature weighting scheme.
        Uses parallel computing of fingerprint pairs.

        Parameters
        ----------
        feature_distances_generator : kinsim_structure.similarity.FeatureDistancesGenerator
            Feature distances for multiple fingerprint pairs.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.
        """

        # Set class attributes
        self.distance_measure = feature_distances_generator.distance_measure
        self.feature_weights = feature_weights
        self.molecule_codes = feature_distances_generator.molecule_codes
        self.kinase_names = feature_distances_generator.kinase_names

        # Calculate pairwise fingerprint distances
        fingerprint_distance_list = self._get_fingerprint_distance_from_list(
            self._get_fingerprint_distance,
            list(feature_distances_generator.data.values()),
            self.feature_weights
        )

        # Format result and save to class attribute
        self.data = pd.DataFrame(
            [
                [i.molecule_codes[0], i.molecule_codes[1], i.data.distance, i.data.coverage]
                for i in fingerprint_distance_list
            ],
            columns='molecule_code_1 molecule_code_2 distance coverage'.split()
        )

    @staticmethod
    def _get_fingerprint_distance_from_list(
            method_get_fingerprint_distance, feature_distances_list, feature_weights=None
    ):
        """
        Get fingerprint distances based on multiple feature distances (i.e. for multiple fingerprint pairs).
        Method uses parallel computing.

        Parameters
        ----------
        method_get_fingerprint_distance : method
            Method calculating fingerprint distance for one fingerprint pair (based on their feature distances).
        feature_distances_list : list of kinsim_structure.similarity.FeatureDistances
            List of distances between two fingerprints for each of their features, plus details on feature type,
            feature, feature bit coverage, and feature bit number.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        list of kinsim_structure.similarity.FingerprintDistance
            List of distance between two fingerprints, plus details on molecule codes, feature weights and feature
            coverage.
        """

        # Get start time of computation
        start = datetime.datetime.now()
        logger.info(f'Calculate pairwise fingerprint distances...')

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f'Number of cores used: {num_cores}')

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Apply function to each chunk in list
        fingerprint_distances_list = pool.starmap(
            method_get_fingerprint_distance,
            zip(feature_distances_list, repeat(feature_weights))
        )

        # Close and join pool
        pool.close()
        pool.join()

        # Get end time of computation
        logger.info(f'Number of fingerprint distances: {len(fingerprint_distances_list)}')
        end = datetime.datetime.now()

        logger.info(f'Start: {start}')
        logger.info(f'End: {end}')

        return fingerprint_distances_list

    @staticmethod
    def _get_fingerprint_distance(feature_distances, feature_weights=None):
        """
        Get the fingerprint distance for one fingerprint pair.

        Parameters
        ----------
        feature_distances : kinsim_structure.similarity.FeatureDistances
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        kinsim_structure.similarity.FingerprintDistance
            Distance between two fingerprints, plus details on molecule codes, feature weights and feature coverage.
        """

        fingerprint_distance = FingerprintDistance()
        fingerprint_distance.from_feature_distances(feature_distances, feature_weights)

        return fingerprint_distance

    def get_structure_distance_matrix(self, fill=False):
        """
        Get fingerprint distances for all structure pairs in the form of a matrix (DataFrame).

        Parameters
        ----------
        fill : bool
            Fill or fill not (default) lower triangle of distance matrix.

        Returns
        -------
        pandas.DataFrame
            Structure distance matrix.
        """

        # Initialize matrix
        structure_distance_matrix = pd.DataFrame(
            [],
            columns=self.molecule_codes,
            index=self.molecule_codes,
            dtype=float
        )

        # Fill matrix with distance values
        for index, row in self.data.iterrows():
            structure_distance_matrix.loc[row.molecule_code_1, row.molecule_code_2] = row.distance

            if fill:
                structure_distance_matrix.loc[row.molecule_code_2, row.molecule_code_1] = row.distance

        # Fill values on matrix main diagonal to 0.0
        for molecule_code in self.molecule_codes:
            structure_distance_matrix.loc[molecule_code, molecule_code] = 0.0

        return structure_distance_matrix

    def get_kinase_distance_matrix(self, by='minimum', fill=False):
        """
        Extract per kinase pair one distance value from the set of structure pair distance values and return these
        fingerprint distances for all kinase pairs in the form of a matrix (DataFrame).

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of distances values per
            structure pair. Default: Minimum distance value.
        fill : bool
            Fill or fill not (default) lower triangle of distance matrix.

        Returns
        -------
        pandas.DataFrame
            Kinase distance matrix.
        """

        # Initialize matrix
        kinase_distance_matrix = pd.DataFrame(
            [],
            columns=self.kinase_names,
            index=self.kinase_names,
            dtype=float
        )

        # Fill matrix with distance values
        for index, row in self._get_kinase_distances(by).iterrows():
            kinase_distance_matrix.loc[index[0], index[1]] = row.distance

            if fill:
                kinase_distance_matrix.loc[index[1], index[0]] = row.distance

        # Fill values on matrix main diagonal to 0.0 which are NaN (i.e. kinases that have only one structure
        # representative)
        for kinase_name in self.kinase_names:
            if np.isnan(kinase_distance_matrix.loc[kinase_name, kinase_name]):
                kinase_distance_matrix.loc[kinase_name, kinase_name] = 0.0

        return kinase_distance_matrix

    def _get_kinase_distances(self, by='minimum'):
        """
        Extract per kinase pair one distance value from the set of structure pair distance values.

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of distances values per
            structure pair. Default: Minimum distance value.

        Returns
        -------
        pandas.DataFrame
            Fingerprint distance and coverage for kinase pairs.
        """

        # Get distance values for structure pairs
        structure_distances = self._add_kinases_to_fingerprint_distance()

        # Group by kinase names
        structure_distances_grouped_by_kinases = structure_distances.groupby(
            by=['kinase_1', 'kinase_2'],
            sort=False
        )

        # Get distance values per kinase pair based on given condition
        by_terms = 'minimum maximum mean'.split()

        if by == 'minimum':
            kinase_distances = structure_distances_grouped_by_kinases.min()
        elif by == 'maximum':
            kinase_distances = structure_distances_grouped_by_kinases.max()
        elif by == 'mean':
            kinase_distances = structure_distances_grouped_by_kinases.mean()
        else:
            raise ValueError(f'Condition "by" unknown. Choose from: {", ".join(by_terms)}')

        return kinase_distances

    def _add_kinases_to_fingerprint_distance(self):
        """
        Add two columns to fingerprint distances for kinase 1 name and kinase 2 name.

        Returns
        -------
        pandas.DataFrame
            Fingerprint distance and coverage, plus details on both molecule codes and kinase names associated with
            fingerprint pairs.
        """

        # Make a copy of distance values per structure pairs
        fingerprint_distance = self.data.copy()

        # Add columns for kinase names (kinase pair)
        fingerprint_distance['kinase_1'] = [
            i.split('/')[1].split('_')[0] for i in fingerprint_distance.molecule_code_1
        ]
        fingerprint_distance['kinase_2'] = [
            i.split('/')[1].split('_')[0] for i in fingerprint_distance.molecule_code_2
        ]

        return fingerprint_distance


class FeatureDistancesGenerator:
    """
    Generate feature distances for multiple fingerprint pairs, given a distance measure.
    Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    distance_measure : str
        Type of distance measure, defaults to scaled Euclidean distance.
    molecule_codes : list of str
        Unique molecule codes associated with all fingerprints (sorted alphabetically).
    kinase_names : list of str
        Unique kinase names associated with all fingerprints (sorted alphabetically).
    data : dict of kinsim_structure.similarity.FeatureDistances
        Dictionary of distances between two fingerprints for each of their features, plus details on feature type,
        feature, feature bit coverage, and feature bit number. Dictionary key is molecule code tuple associated with
        fingerprint pair.
    """

    def __init__(self):

        self.distance_measure = None
        self.molecule_codes = None
        self.kinase_names = None
        self.data = None

    def from_fingerprint_generator(self, fingerprints_generator, distance_measure='scaled_euclidean'):
        """
        Calculate feature distances for all possible fingerprint pair combinations, given a distance measure.

        Parameters
        ----------
        fingerprints_generator : kinsim_structure.encoding.FingerprintsGenerator
            Multiple fingerprints.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.
        """

        # Remove empty fingerprints
        fingerprints = self._remove_empty_fingerprints(fingerprints_generator.data)

        # Set class attributes
        self.distance_measure = distance_measure
        self.molecule_codes = sorted(fingerprints_generator.data.keys())
        self.kinase_names = sorted(set([i.split('/')[1].split('_')[0] for i in self.molecule_codes]))

        # Get fingerprint pairs (molecule code pairs)
        pairs = self._get_fingerprint_pairs(fingerprints)

        # Calculate pairwise feature distances
        feature_distances_list = self._get_feature_distances_from_list(
            self._get_feature_distances,
            pairs,
            fingerprints,
            self.distance_measure
        )

        # Cast both attributes to DataFrames
        self.data = {
            (i.molecule_codes[0], i.molecule_codes[1]): i for i in feature_distances_list
        }

    @staticmethod
    def _get_feature_distances_from_list(
            method_get_feature_distances, pairs, fingerprints, distance_measure='scaled_euclidean'
    ):
        """
        Get feature distances for multiple fingerprint pairs.
        Method uses parallel computing.

        Parameters
        ----------
        method_get_feature_distances : method
            Method calculating feature distances for one fingerprint pair.
        fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        pairs : list of list of str
            List of molecule code pairs (list).
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.

        Returns
        -------
        list of kinsim_structure.similarity.FeatureDistances
            List of distances between two fingerprints for each of their features, plus details on feature type,
            feature, feature bit coverage, and feature bit number.
        """

        # Get start time of computation
        start = datetime.datetime.now()
        logger.info(f'Calculate pairwise feature distances...')

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f'Number of cores used: {num_cores}')

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Apply function to each chunk in list
        feature_distances_list = pool.starmap(
            method_get_feature_distances,
            zip(pairs, repeat(fingerprints), repeat(distance_measure))
        )

        # Close and join pool
        pool.close()
        pool.join()

        # Get end time of script
        logger.info(f'Number of feature distances: {len(feature_distances_list)}')
        end = datetime.datetime.now()

        logger.info(start)
        logger.info(end)

        return feature_distances_list

    @staticmethod
    def _get_feature_distances(pair, fingerprints, distance_measure='scaled_euclidean'):
        """
        Calculate the feature distances for one fingerprint pair.

        Parameters
        ----------
        fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        pair : list of str
            Molecule names of molecules encoded by fingerprint pair.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.

        Returns
        -------
        kinsim_structure.similarity.FeatureDistances
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        """

        fingerprint1 = fingerprints[pair[0]]
        fingerprint2 = fingerprints[pair[1]]

        feature_distances = FeatureDistances()
        feature_distances.from_fingerprints(fingerprint1, fingerprint2, distance_measure)

        return feature_distances

    @staticmethod
    def _get_fingerprint_pairs(fingerprints):
        """
        Get all fingerprint pair combinations from dictionary of fingerprints.

        Parameters
        ----------
        fingerprints : dict of kinsim_structure.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.

        Returns
        -------
        list of list of str
            List of molecule code pairs (list).
        """

        pairs = []

        for i, j in combinations(fingerprints.keys(), 2):
            pairs.append([i, j])

        logger.info(f'Number of pairs: {len(pairs)}')

        return pairs

    @staticmethod
    def _remove_empty_fingerprints(fingerprints):
        """
        Remove empty fingerprints from dictionary of fingerprints.

        Parameters
        ----------
        fingerprints : dict of (kinsim_structure.encoding.Fingerprint or None)
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.

        Returns
        -------
        dict of kinsim_structure.encoding.Fingerprint
            Dictionary of non-empty fingerprints: Keys are molecule codes and values are fingerprint data.
        """

        # Get molecule codes for empty fingerprints
        empty_molecule_codes = []

        for molecule_code, fingerprint in fingerprints.items():

            if not fingerprint:
                empty_molecule_codes.append(molecule_code)
                logger.info(f'Empty fingerprint molecule codes: {molecule_code}')

        # Delete empty fingerprints from dict
        for empty in empty_molecule_codes:
            del fingerprints[empty]

        logger.info(f'Number of empty fingerprints: {len(empty_molecule_codes)}')
        logger.info(f'Number of non-empty fingerprints: {len(fingerprints)}')

        return fingerprints


class FingerprintDistance:
    """
    Distance between two fingerprints using feature-wise weighting.

    Attributes
    ----------
    molecule_codes : list of str
        Codes of both molecules represented by the fingerprints.
    distance_measure : str
        Type of distance measure, defaults to scaled Euclidean distance.
    feature_weights : dict of floats
        Weights per feature.
    data : pandas.Series
        Fingerprint distance and coverage (weighted per feature).
    """

    def __init__(self):

        self.molecule_codes = None
        self.distance_measure = None
        self.feature_weights = None
        self.data = None

    def from_feature_distances(self, feature_distances, feature_weights=None):
        """
        Get fingerprint distance.

        Parameters
        ----------
        feature_distances : kinsim_structure.similarity.FeatureDistances
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        float
            Fingerprint distance.
        """

        # Set class attributes
        self.molecule_codes = feature_distances.molecule_codes
        self.distance_measure = feature_distances.distance_measure

        # Get feature distances data
        feature_distances = feature_distances.data

        # Add weights
        feature_distances = self._add_weight_column(feature_distances, feature_weights)
        self.feature_weights = feature_distances.weight

        # Calculate weighted sum of feature distances and feature coverage
        fingerprint_distance = (feature_distances.distance * feature_distances.weight).sum()
        fingerprint_coverage = (feature_distances.bit_coverage * feature_distances.weight).sum()

        # Format results and save to class attribute
        self.data = pd.Series(
            [fingerprint_distance, fingerprint_coverage],
            index='distance coverage'.split()
        )

    def _add_weight_column(self, feature_distances, feature_weights=None):
        """
        Add feature weights to feature distance details (each feature or feature type can be set individually).

        Parameters
        ----------
        feature_distances : pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        feature_weights : dict of float or None
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type
                Feature types to be set are: physicochemical, distances, and moments.
            (iii) By feature:
                Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, feature bit number, AND feature weights.
        """

        # The parameter feature_weights can come in three difference formats as described in this method's docstring.
        # For each of the three formats perform a certain action:

        if feature_weights is None:

            # Defaults to equally distributed weights between all features
            feature_weights = self._format_weight_per_feature(feature_weights)
            return pd.merge(feature_distances, feature_weights, on='feature_name', sort=False)

        elif isinstance(feature_weights, dict):

            # Try to figure out if input feature weights are per feature or feature type

            if len(feature_weights) <= 3:

                # Set weights per feature type
                feature_weights = self._format_weight_per_feature_type(feature_weights)
                return pd.merge(feature_distances, feature_weights, on='feature_name', sort=False)

            else:

                # Set weights per feature
                feature_weights = self._format_weight_per_feature(feature_weights)
                return pd.merge(feature_distances, feature_weights, on='feature_name', sort=False)

        else:

            raise TypeError(f'Data type of "feature_weights" parameter must be dict, but is {type(feature_weights)}.')

    @staticmethod
    def _format_weight_per_feature_type(feature_type_weights=None):
        """
        Distribute feature type weights equally to features per feature type and format these values as a DataFrame
        with 15 rows (features) and 2 columns (feature name, weight).

        Parameters
        ----------
        feature_type_weights : dict of float (3 items) or None
            Weights per feature type which need to sum up to 1.0.
            Feature types to be set are: physicochemical, distances, and moments.
            Default feature weights (None) are set equally distributed to 1/3 (3 feature types in total).

        Returns
        -------
        pandas.DataFrame
            Feature weights: 15 rows (features) and 2 columns (feature name, weight).
        """

        # 1. Either set feature weights to default or check if non-default input is correct
        equal_weights = 1.0 / 3

        feature_type_weights_default = {
            'physicochemical': equal_weights,
            'distances': equal_weights,
            'moments': equal_weights
        }

        if feature_type_weights is None:

            feature_type_weights = feature_type_weights_default

        else:

            # Check data type of feature weights
            if not isinstance(feature_type_weights, dict):
                raise TypeError(f'Data type of "feature_weights" parameter must be dict, but is '
                                f'{type(feature_type_weights)}.')

            # Check if feature weight keys are correct
            if not feature_type_weights.keys() == feature_type_weights_default.keys():
                raise ValueError(f'Feature weights contain unknown or missing feature(s). Set the following features: '
                                 f'{", ".join(list(feature_type_weights_default.keys()))}.')

            # Check if feature weight values are correct
            for feature_name, weight in feature_type_weights.items():
                if not isinstance(weight, float):
                    raise TypeError(f'Weight for feature "{feature_name}" must be float, but is {type(weight)}.')

            # Check if sum of weights is 1.0
            if sum(feature_type_weights.values()) != 1.0:
                raise ValueError(f'Sum of all weights must be one, but is {sum(feature_type_weights.values())}.')

        # 2. Distribute feature type weight equally to features in feature type
        feature_weights = {}

        for feature_type, feature_names in FEATURE_NAMES.items():

            weight_per_feature_in_feature_type = feature_type_weights[feature_type] / len(feature_names)

            for feature_name in feature_names:
                feature_weights[feature_name] = weight_per_feature_in_feature_type

        # 3. Get feature weights as DataFrame with feature names
        feature_weights = pd.DataFrame.from_dict(feature_weights, orient='index', columns=['weight'])
        feature_weights['feature_name'] = feature_weights.index
        feature_weights.reset_index(inplace=True, drop=True)

        return feature_weights

    @staticmethod
    def _format_weight_per_feature(feature_weights=None):
        """
        Format input feature weights to DataFrame with 15 rows (features) and 2 columns (feature name, weight).

        Parameters
        ----------
        feature_weights : dict of float (15 items) or None
            Weights per feature which need to sum up to 1.0.
            Features to be set are: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure, distance_to_centroid,
            distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            Default feature weights (None) are set equally distributed to 1/15 (15 features in total).

        Returns
        -------
        pandas.DataFrame
            Feature weights: 15 rows (features) and 2 columns (feature name, weight).
        """

        # 1. Either set feature weights to default or check if non-default input is correct
        equal_weights = 1.0 / 15

        feature_weights_default = {
            'size': equal_weights,
            'hbd': equal_weights,
            'hba': equal_weights,
            'charge': equal_weights,
            'aromatic': equal_weights,
            'aliphatic': equal_weights,
            'sco': equal_weights,
            'exposure': equal_weights,
            'distance_to_centroid': equal_weights,
            'distance_to_hinge_region': equal_weights,
            'distance_to_dfg_region': equal_weights,
            'distance_to_front_pocket': equal_weights,
            'moment1': equal_weights,
            'moment2': equal_weights,
            'moment3': equal_weights
        }

        if feature_weights is None:

            feature_weights = feature_weights_default

        else:

            # Check data type of feature weights
            if not isinstance(feature_weights, dict):
                raise TypeError(f'Data type of "feature_weights" parameter must be dict, but is '
                                f'{type(feature_weights)}.')

            # Check if feature weight keys are correct
            if not feature_weights.keys() == feature_weights_default.keys():
                raise ValueError(f'Feature weights contain unknown or missing feature(s). Set the following features: '
                                 f'{", ".join(list(feature_weights_default.keys()))}.')

            # Check if feature weight values are correct
            for feature_name, weight in feature_weights.items():
                if not isinstance(weight, float):
                    raise TypeError(f'Weight for feature "{feature_name}" must be float, but is {type(weight)}.')

            # Check if sum of weights is 1.0
            if sum(feature_weights.values()) != 1.0:
                raise ValueError(f'Sum of all weights must be one, but is {sum(feature_weights.values())}.')

        # 2. Get feature weights as DataFrame with feature names
        feature_weights = pd.DataFrame.from_dict(feature_weights, orient='index', columns=['weight'])
        feature_weights['feature_name'] = feature_weights.index
        feature_weights.reset_index(inplace=True, drop=True)

        return feature_weights


class FeatureDistances:
    """
    Distances between two fingerprints for each of their features, plus details on feature type, feature,
    feature bit coverage, and feature bit number.

    Attributes
    ----------
    molecule_codes : list of str
        Codes of both molecules represented by the fingerprints.
    distance_measure : str
        Type of distance measure, defaults to scaled Euclidean distance.
    data : pandas.DataFrame
        Distances between two fingerprints for each of their features, plus details on feature type, feature,
        feature bit coverage, and feature bit number.
    """

    def __init__(self):

        self.molecule_codes = None
        self.distance_measure = None
        self.data = None

    def from_fingerprints(self, fingerprint1, fingerprint2, distance_measure='scaled_euclidean'):
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

        Returns
        -------
        pandas.DataFrame
            Distances between two fingerprints for each of their features, plus details on feature type, feature,
            feature bit coverage, and feature bit number.
        """

        # Set class attributes
        self.molecule_codes = [fingerprint1.molecule_code, fingerprint2.molecule_code]
        self.distance_measure = distance_measure

        # Get fingerprint pair (normalized fingerprints only)
        fingerprint_pair = self._extract_fingerprint_pair(fingerprint1, fingerprint2, normalized=True)

        feature_distances = []

        for feature_type in FEATURE_NAMES.keys():

            for feature_name in FEATURE_NAMES[feature_type]:

                # Get feature distance
                feature_distance = self._calculate_feature_distance(
                        fingerprint_pair[feature_type][feature_name],
                        distance_measure
                    )

                # Get number of feature bits without any NaN value
                bit_number = len(fingerprint_pair[feature_type][feature_name])

                # Get bit coverage
                bit_coverage = self._get_bit_coverage(feature_type, bit_number)

                # Save feature data to fingerprint data
                feature_distances.append([feature_type, feature_name, feature_distance, bit_coverage, bit_number])

        # Format result and save to class attribute
        self.data = pd.DataFrame(
            feature_distances,
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

        # Define number of bits per feature type
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

    def _calculate_feature_distance(self, feature_pair, distance_measure='scaled_euclidean'):
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

        # Test if parameter input is correct
        if not isinstance(feature_pair, pd.DataFrame):
            raise TypeError(f'Parameter "feature_pair" must be of type pandas.DataFrame, but is {type(feature_pair)}.')

        if feature_pair.shape[1] != 2:
            raise ValueError(f'Parameter "feature_pair" must be pandas.DataFrame with two columns, '
                             f'but has {feature_pair.shape[1]} columns.')

        # Set feature distance to NaN if no bits available for distance calculation
        if len(feature_pair) == 0:
            return np.nan

        # In case there are NaN positions in input, remove bit positions containing any NaN value
        feature_pair.dropna(how='any', axis=0, inplace=True)

        # Get feature distance
        if distance_measure == 'scaled_euclidean':
            return self._scaled_euclidean_distance(
                feature_pair.iloc[:, 0],  # Fingerprint 1
                feature_pair.iloc[:, 1]  # Fingerprint 2
            )

        elif distance_measure == 'scaled_cityblock':
            return self._scaled_cityblock_distance(
                feature_pair.iloc[:, 0],  # Fingerprint 1
                feature_pair.iloc[:, 1]  # Fingerprint 2
            )

        else:
            distance_measures = 'scaled_euclidean scaled_cityblock'.split()
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
    def _scaled_euclidean_distance(values1, values2):
        """
        Calculate scaled Euclidean distance between two value lists of same length.

        Parameters
        ----------
        values1 : list or pandas.Series
            Value list (same length as values2).
        values2 : list or pandas.Series
            Value list (same length as values1).

        Returns
        -------
        float
            Scaled Euclidean distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f'Distance calculation failed: Values lists are not of same length.')
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
        values1 : list or pandas.Series
            Value list (same length as values2).
        values2 : list or pandas.Series
            Value list (same length as values1).

        Returns
        -------
        float
            Scaled cityblock distance between two value lists.
        """

        if len(values1) != len(values2):
            raise ValueError(f'Distance calculation failed: Values lists are not of same length.')
        elif len(values1) == 0:
            return np.nan
        else:
            d = 1 / len(values1) * distance.cityblock(values1, values2)
            return d
