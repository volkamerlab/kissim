"""
similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Handles the primary functions for the structural kinase fingerprint comparison.
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool

from itertools import combinations, repeat, chain
import numpy as np
import pandas as pd
from scipy.spatial import distance

from kissim.encoding import FEATURE_NAMES

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
    feature_weights : None or list of float
        Feature weights of the following form:
        (i) None
            Default feature weights: All features equally distributed to 1/15 (15 features in total).
        (ii) By feature type (list of 3 floats)
            Feature types to be set in the following order: physicochemical, distances, and moments.
        (iii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
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
        feature_distances_generator : kissim.similarity.FeatureDistancesGenerator
            Feature distances for multiple fingerprint pairs.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.
        """

        start = datetime.datetime.now()

        logger.info(f"SIMILARITY: FingerprintDistanceGenerator: {feature_weights}")

        # Set class attributes
        self.distance_measure = feature_distances_generator.distance_measure
        self.feature_weights = feature_weights
        self.molecule_codes = feature_distances_generator.molecule_codes
        self.kinase_names = feature_distances_generator.kinase_names

        # Calculate pairwise fingerprint distances
        fingerprint_distance_list = self._get_fingerprint_distance_from_list(
            self._get_fingerprint_distance,
            list(feature_distances_generator.data.values()),
            self.feature_weights,
        )

        # Format result and save to class attribute
        self.data = pd.DataFrame(
            [
                [i.molecule_pair_code[0], i.molecule_pair_code[1], i.distance, i.bit_coverage]
                for i in fingerprint_distance_list
            ],
            columns="molecule_code_1 molecule_code_2 distance coverage".split(),
        )

        end = datetime.datetime.now()

        logger.info(f"Start of fingerprint distance generation: {start}")
        logger.info(f"End of fingerprint distance generation: {end}")

    @staticmethod
    def _get_fingerprint_distance_from_list(
        _get_fingerprint_distance, feature_distances_list, feature_weights=None
    ):
        """
        Get fingerprint distances based on multiple feature distances (i.e. for multiple fingerprint pairs).
        Uses parallel computing.

        Parameters
        ----------
        _get_fingerprint_distance : method
            Method calculating fingerprint distance for one fingerprint pair (based on their feature distances).
        feature_distances_list : list of kissim.similarity.FeatureDistances
            List of distances and bit coverages between two fingerprints for each of their features.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        list of kissim.similarity.FingerprintDistance
            List of distance between two fingerprints, plus details on molecule codes, feature weights and feature
            coverage.
        """

        # Get start time of computation
        start = datetime.datetime.now()
        logger.info(f"Calculate pairwise fingerprint distances...")

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Apply function to each chunk in list
        fingerprint_distances_list = pool.starmap(
            _get_fingerprint_distance, zip(feature_distances_list, repeat(feature_weights))
        )

        # Close and join pool
        pool.close()
        pool.join()

        # Get end time of computation
        logger.info(f"Number of fingerprint distances: {len(fingerprint_distances_list)}")
        end = datetime.datetime.now()

        logger.info(f"Start: {start}")
        logger.info(f"End: {end}")

        return fingerprint_distances_list

    @staticmethod
    def _get_fingerprint_distance(feature_distances, feature_weights=None):
        """
        Get the fingerprint distance for one fingerprint pair.

        Parameters
        ----------
        feature_distances : kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        kissim.similarity.FingerprintDistance
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
            [], columns=self.molecule_codes, index=self.molecule_codes, dtype=float
        )

        # Fill matrix with distance values
        for index, row in self.data.iterrows():
            structure_distance_matrix.loc[row.molecule_code_1, row.molecule_code_2] = row.distance

            if fill:
                structure_distance_matrix.loc[
                    row.molecule_code_2, row.molecule_code_1
                ] = row.distance

        # Fill values on matrix main diagonal to 0.0
        for molecule_code in self.molecule_codes:
            structure_distance_matrix.loc[molecule_code, molecule_code] = 0.0

        return structure_distance_matrix

    def get_kinase_distance_matrix(self, by="minimum", fill=False):
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
            [], columns=self.kinase_names, index=self.kinase_names, dtype=float
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

    def _get_kinase_distances(self, by="minimum"):
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
            by=["kinase_1", "kinase_2"], sort=False
        )

        # Get distance values per kinase pair based on given condition
        by_terms = "minimum maximum mean size".split()

        if by == "minimum":
            kinase_distances = structure_distances_grouped_by_kinases.min()
        elif by == "maximum":
            kinase_distances = structure_distances_grouped_by_kinases.max()
        elif by == "mean":
            kinase_distances = structure_distances_grouped_by_kinases.mean()
        elif by == "size":
            kinase_distances = structure_distances_grouped_by_kinases.size()
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
        fingerprint_distance["kinase_1"] = [
            i.split("/")[1].split("_")[0] for i in fingerprint_distance.molecule_code_1
        ]
        fingerprint_distance["kinase_2"] = [
            i.split("/")[1].split("_")[0] for i in fingerprint_distance.molecule_code_2
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
    data : dict of tuple of str: np.ndarray
        Feature distances and bit coverage (value) for each fingerprint pair (key: molecule codes).
    """

    def __init__(self):

        self.distance_measure = None
        self.data = None

    @property
    def molecule_codes(self):
        """
        Unique molecule codes associated with all fingerprints (sorted alphabetically).

        Returns
        -------
        list of str:
            Molecule codes.
        """

        if self.data is not None:
            return sorted(list(set(chain.from_iterable(self.data.keys()))))

    @property
    def kinase_names(self):
        """
        Unique kinase names associated with all fingerprints (sorted alphabetically).

        Returns
        -------
        list of str
            Kinase names.
        """

        if self.molecule_codes is not None:
            return sorted(set([i.split("/")[1].split("_")[0] for i in self.molecule_codes]))

    def from_fingerprint_generator(
        self, fingerprints_generator, distance_measure="scaled_euclidean"
    ):
        """
        Calculate feature distances for all possible fingerprint pair combinations, given a distance measure.

        Parameters
        ----------
        fingerprints_generator : kissim.encoding.FingerprintsGenerator
            Multiple fingerprints.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.
        """

        start = datetime.datetime.now()

        logger.info(f"SIMILARITY: FeatureDistancesGenerator: {distance_measure}")

        # Remove empty fingerprints
        fingerprints = self._remove_empty_fingerprints(fingerprints_generator.data)

        # Set class attributes
        self.distance_measure = distance_measure

        # Calculate pairwise feature distances
        feature_distances_list = self._get_feature_distances_from_list(
            self._get_feature_distances, fingerprints, self.distance_measure
        )

        # Cast returned list into dict
        self.data = {i.molecule_pair_code: i for i in feature_distances_list}

        end = datetime.datetime.now()

        logger.info(f"Start of feature distances generator: {start}")
        logger.info(f"End of feature distances generator: {end}")

    def get_data_by_molecule_pair(self, molecule_code1, molecule_code2):
        """
        Get feature distances for fingerprint pair by their molecule codes, with details on feature types,
        feature names, and feature bit coverages.

        Parameters
        ----------
        molecule_code1 : str
            Molecule code 1.
        molecule_code2 : str
            Molecule code 2.

        Returns
        -------
        pandas.DataFrame
            Feature distances for fingerprint pair with details on feature types, features names, and feature bit
            coverages.
        """

        if self.data is not None:

            feature_types = list(
                chain.from_iterable([[key] * len(value) for key, value in FEATURE_NAMES.items()])
            )
            feature_names = list(chain.from_iterable(FEATURE_NAMES.values()))

            data = self.data[(molecule_code1, molecule_code2)]

            data_df = pd.DataFrame(data, columns="distance bit_coverage".split())
            data_df.insert(loc=0, column="feature_type", value=feature_types)
            data_df.insert(loc=1, column="feature_names", value=feature_names)

            return data_df

    def _get_feature_distances_from_list(
        self, _get_feature_distances, fingerprints, distance_measure="scaled_euclidean"
    ):
        """
        Get feature distances for multiple fingerprint pairs.
        Uses parallel computing.

        Parameters
        ----------
        _get_feature_distances : method
            Method calculating feature distances for one fingerprint pair.
        fingerprints : dict of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.

        Returns
        -------
        list of kissim.similarity.FeatureDistances
            List of distances and bit coverages between two fingerprints for each of their features.
        """

        # Get start time of computation
        start = datetime.datetime.now()
        logger.info(f"Calculate pairwise feature distances...")

        # Number of CPUs on machine
        num_cores = cpu_count() - 1
        logger.info(f"Number of cores used: {num_cores}")

        # Create pool with `num_processes` processes
        pool = Pool(processes=num_cores)

        # Get fingerprint pairs (molecule code pairs)
        pairs = self._get_fingerprint_pairs(fingerprints)

        # Apply function to each chunk in list
        feature_distances_list = pool.starmap(
            _get_feature_distances, zip(pairs, repeat(fingerprints), repeat(distance_measure))
        )

        # Close and join pool
        pool.close()
        pool.join()

        # Get end time of script
        logger.info(f"Number of feature distances: {len(feature_distances_list)}")
        end = datetime.datetime.now()

        logger.info(start)
        logger.info(end)

        return feature_distances_list

    @staticmethod
    def _get_feature_distances(pair, fingerprints, distance_measure="scaled_euclidean"):
        """
        Calculate the feature distances for one fingerprint pair.

        Parameters
        ----------
        fingerprints : dict of tuple of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        pair : tuple of str
            Molecule names of molecules encoded by fingerprint pair.
        distance_measure : str
            Type of distance measure, defaults to scaled Euclidean distance.

        Returns
        -------
        kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
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
        fingerprints : dict of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.

        Returns
        -------
        list of tuple of str
            List of molecule code pairs (list).
        """

        pairs = []

        for i, j in combinations(fingerprints.keys(), 2):
            pairs.append((i, j))

        logger.info(f"Number of pairs: {len(pairs)}")

        return pairs

    @staticmethod
    def _remove_empty_fingerprints(fingerprints):
        """
        Remove empty fingerprints from dictionary of fingerprints.

        Parameters
        ----------
        fingerprints : dict of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.

        Returns
        -------
        dict of str: kissim.encoding.Fingerprint
            Dictionary of non-empty fingerprints: Keys are molecule codes and values are fingerprint data.
        """

        # Get molecule codes for empty fingerprints
        empty_molecule_codes = []

        for molecule_code, fingerprint in fingerprints.items():

            if not fingerprint:
                empty_molecule_codes.append(molecule_code)
                logger.info(f"Empty fingerprint molecule codes: {molecule_code}")

        # Delete empty fingerprints from dict
        for empty in empty_molecule_codes:
            del fingerprints[empty]

        logger.info(f"Number of empty fingerprints: {len(empty_molecule_codes)}")
        logger.info(f"Number of non-empty fingerprints: {len(fingerprints)}")

        return fingerprints


class FingerprintDistance:
    """
    Distance between two fingerprints using feature-wise weighting.

    Attributes
    ----------
    molecule_pair_code : tuple of str
        Codes of both molecules represented by the fingerprints.
    distance : float
        Fingerprint distance (weighted per feature).
    bit_coverage : float
        Fingerprint coverage (weighted per feature).
    """

    def __init__(self):

        self.molecule_pair_code = None
        self.distance = None
        self.bit_coverage = None

    def from_feature_distances(self, feature_distances, feature_weights=None):
        """
        Get fingerprint distance.

        Parameters
        ----------
        feature_distances : kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.
        """

        # Set class attributes
        self.molecule_pair_code = feature_distances.molecule_pair_code

        # Add weights
        feature_weights_formatted = self._format_weights(feature_weights)

        # Calculate weighted sum of feature distances and feature coverage
        self.distance = sum(feature_distances.distances * feature_weights_formatted)
        self.bit_coverage = sum(feature_distances.bit_coverages * feature_weights_formatted)

    def _format_weights(self, feature_weights=None):
        """
        Get feature weights based on input weights (each feature or feature type can be set individually).

        Parameters
        ----------
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        np.ndarray
            Feature weights.
        """

        # The parameter feature_weights can come in three difference formats as described in this method's docstring.
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
                f'Data type of "feature_weights" parameter must be list, but is {type(feature_weights)}.'
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
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and moments.
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

        # 2. Distribute feature type weight equally to features in feature type (in default feature order)
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
                Default feature weights: All features equally distributed to 1/15 (15 features in total).
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic, sco, exposure,
                distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region, distance_to_front_pocket,
                moment1, moment2, and moment3.
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


class FeatureDistances:
    """
    Distances between two fingerprints for each of their features, plus details on feature type, feature,
    feature bit coverage, and feature bit number.

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
        Feature distances for fingerprint pair, with details on feature types, feature names, and feature bit coverages.

        Returns
        -------
        pandas.DataFrame
            Feature distances for fingerprint pair with details on feature types, features names, and feature bit
            coverages.
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

    def from_fingerprints(
        self, fingerprint1, fingerprint2, distance_measure="scaled_euclidean", normalized=True
    ):
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
        normalized : bool
            Normalized (default) or non-normalized fingerprints.
        """

        # Set class attributes
        self.molecule_pair_code = (fingerprint1.molecule_code, fingerprint2.molecule_code)

        # Get fingerprint (normalized or not normalized)
        if normalized:
            f1 = fingerprint1.fingerprint_normalized
            f2 = fingerprint2.fingerprint_normalized
        else:
            f1 = fingerprint1.fingerprint
            f2 = fingerprint2.fingerprint

        # Iterate over all features and get feature type, feature name, feature distance and feature bit coverage
        distances = []
        bit_coverages = []

        for feature_type in FEATURE_NAMES.keys():

            for feature_name in FEATURE_NAMES[feature_type]:

                # Get feature bits
                features1 = f1[feature_type][feature_name]
                features2 = f2[feature_type][feature_name]

                distance, bit_coverage = self.from_features(features1, features2, distance_measure)

                # Save feature data to fingerprint data
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
            Pairwise bits of one feature extracted from two fingerprints (only bit positions without any NaN value).
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
                f'Parameter "feature_pair" has not two (i.e. {feature_pair.shape[1]}) np.ndarray rows.'
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


if __name__ == "__main__":
    print("similarity.py executed from CLI.")
