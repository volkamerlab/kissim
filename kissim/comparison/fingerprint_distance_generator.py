"""
kissim.comparison.fingerprint_distance_generator

Defines the pairwise fingerprint distances for a set of fingerprints.
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool

import numpy as np
import pandas as pd

from . import FingerprintDistance

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
            Default feature weights: All features equally distributed to 1/15
            (15 features in total).
        (ii) By feature type (list of 3 floats)
            Feature types to be set in the following order: physicochemical, distances, and
            moments.
        (iii) By feature (list of 15 floats):
            Features to be set in the following order: size, hbd, hba, charge, aromatic, aliphatic,
            sco, exposure,
            distance_to_centroid, distance_to_hinge_region, distance_to_dfg_region,
            distance_to_front_pocket, moment1, moment2, and moment3.
        For (ii) and (iii): All floats must sum up to 1.0.
    data : pandas.DataFrame
        Fingerprint distance and coverage, plus details on both molecule codes associated with
        fingerprint pairs.
    """

    def __init__(self):

        self.distance_measure = None
        self.feature_weights = None
        self.molecule_codes = None
        self.kinase_names = None
        self.data = None

    def from_feature_distances_generator(self, feature_distances_generator, feature_weights=None):
        """
        Generate fingerprint distances for multiple fingerprint pairs based on their feature
        distances, given a feature weighting scheme.
        Uses parallel computing of fingerprint pairs.

        Parameters
        ----------
        feature_distances_generator : kissim.similarity.FeatureDistancesGenerator
            Feature distances for multiple fingerprint pairs.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and
                moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
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
        Get fingerprint distances based on multiple feature distances
        (i.e. for multiple fingerprint pairs).
        Uses parallel computing.

        Parameters
        ----------
        _get_fingerprint_distance : method
            Method calculating fingerprint distance for one fingerprint pair
            (based on their feature distances).
        feature_distances_list : list of kissim.similarity.FeatureDistances
            List of distances and bit coverages between two fingerprints for each of their
            features.
        feature_weights : None or list of float
            Feature weights of the following form:
            (i) None
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and
                moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        list of kissim.similarity.FingerprintDistance
            List of distance between two fingerprints, plus details on molecule codes, feature
            weights and feature coverage.
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
                Default feature weights: All features equally distributed to 1/15
                (15 features in total).
            (ii) By feature type (list of 3 floats)
                Feature types to be set in the following order: physicochemical, distances, and
                moments.
            (iii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
            For (ii) and (iii): All floats must sum up to 1.0.

        Returns
        -------
        kissim.similarity.FingerprintDistance
            Distance between two fingerprints, plus details on molecule codes, feature weights and
            feature coverage.
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
        Extract per kinase pair one distance value from the set of structure pair distance values
        and return these  fingerprint distances for all kinase pairs in the form of a matrix
        (DataFrame).

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.
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

        # Fill values on matrix main diagonal to 0.0 which are NaN
        # (i.e. kinases that have only one structure representative)
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
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.

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
            Fingerprint distance and coverage, plus details on both molecule codes and kinase names
            associated with fingerprint pairs.
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
