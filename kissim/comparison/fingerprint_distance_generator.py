"""
kissim.comparison.fingerprint_distance_generator

Defines the pairwise fingerprint distances for a set of fingerprints.
"""

import datetime
from itertools import repeat
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
    data : pandas.DataFrame
        Fingerprint distance and coverage, plus details on both molecule codes associated with
        fingerprint pairs.
    structure_kinase_ids : list of tuple
        Structure and kinase IDs for structures in dataset.
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
    """

    def __init__(self):

        self.data = None
        self.structure_kinase_ids = None
        self.feature_weights = None

    @classmethod
    def from_feature_distances_generator(cls, feature_distances_generator, feature_weights=None):
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

        Returns
        -------
        kissim.comparison.FingerprintDistanceGenerator
            Fingerprint distance generator.
        """

        fingerprint_distance_generator = cls()

        # Set class attributes
        fingerprint_distance_generator.feature_weights = feature_weights

        # Calculate pairwise fingerprint distances
        fingerprint_distance_list = (
            fingerprint_distance_generator._get_fingerprint_distance_from_list(
                fingerprint_distance_generator._get_fingerprint_distance,
                list(feature_distances_generator.data.values()),
                fingerprint_distance_generator.feature_weights,
            )
        )

        # Format result and save to class attribute
        fingerprint_distance_generator.data = pd.DataFrame(
            [
                [
                    i.structure_pair_ids[0],
                    i.structure_pair_ids[1],
                    i.kinase_pair_ids[0],
                    i.kinase_pair_ids[1],
                    i.distance,
                    i.bit_coverage,
                ]
                for i in fingerprint_distance_list
            ],
            columns="structure1 structure2 kinase1 kinase2 distance coverage".split(),
        )
        fingerprint_distance_generator.structure_kinase_ids = (
            feature_distances_generator.structure_kinase_ids
        )
        return fingerprint_distance_generator

    @classmethod
    def from_structure_klifs_ids(cls, feature_distances_generator, feature_weights=None):
        pass

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

    @property
    def structure_ids(self):
        """
        Unique structure IDs associated with all fingerprints (sorted alphabetically).

        Returns
        -------
        list of str or int
            Structure IDs.
        """

        return sorted(
            pd.DataFrame(self.structure_kinase_ids, columns=["structure_id", "kinase_id"])[
                "structure_id"
            ].unique()
        )

    @property
    def kinase_ids(self):
        """
        Unique kinase IDs (e.g. kinase names) associated with all fingerprints (sorted
        alphabetically).

        Returns
        -------
        list of str or int
            Kinase IDs.
        """

        return sorted(
            pd.DataFrame(self.structure_kinase_ids, columns=["structure_id", "kinase_id"])[
                "kinase_id"
            ].unique()
        )

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

        fingerprint_distance = FingerprintDistance.from_feature_distances(
            feature_distances, feature_weights
        )

        return fingerprint_distance

    def structure_distance_matrix(self):
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

        # Data for upper half of the matrix
        pairs_upper = self.data[["structure1", "structure2", "distance"]]
        # Data for lower half of the matrix
        pairs_lower = pairs_upper.rename(
            columns={"structure1": "structure2", "structure2": "structure1"}
        )

        # Concatenate upper and lower matrix data
        pairs = pd.concat([pairs_upper, pairs_lower]).sort_values(["structure1", "structure2"])
        # Convert to matrix
        matrix = pairs.pivot(columns="structure2", index="structure1", values="distance")
        # Matrix diagonal is NaN > set to 0.0
        matrix = matrix.fillna(0.0)

        return matrix

    def kinase_distance_matrix(self, by="minimum"):
        """
        Extract per kinase pair one distance value from the set of structure pair distance values
        and return these  fingerprint distances for all kinase pairs in the form of a matrix
        (DataFrame).

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.

        Returns
        -------
        pandas.DataFrame
            Kinase distance matrix.
        """

        # Data for upper half of the matrix
        pairs_upper = self.kinase_distances(by).reset_index()[["kinase1", "kinase2", "distance"]]
        # Data for lower half of the matrix
        pairs_lower = pairs_upper.rename(columns={"kinase1": "kinase2", "kinase2": "kinase1"})

        # Concatenate upper and lower matrix data
        pairs = (
            pd.concat([pairs_upper, pairs_lower])
            .sort_values(["kinase1", "kinase2"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Convert to matrix
        matrix = pairs.pivot(columns="kinase2", index="kinase1", values="distance")
        # Matrix diagonal is NaN > set to 0.0
        matrix = matrix.fillna(0.0)

        return matrix

    def kinase_distances(self, by="minimum"):
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

        # Add self-comparisonsa
        data = self.data
        data_self_comparisons = pd.DataFrame(
            [
                [structure_id, structure_id, kinase_id, kinase_id, 0.0, np.nan]
                for (
                    structure_id,
                    kinase_id,
                ) in self.structure_kinase_ids
            ],
            columns=["structure1", "structure2", "kinase1", "kinase2", "distance", "coverage"],
        )
        data = pd.concat([data, data_self_comparisons])

        # Group by kinase names
        structure_distances_grouped_by_kinases = data.groupby(
            by=["kinase1", "kinase2"], sort=False
        )

        # Get distance values per kinase pair based on given condition
        by_terms = "minimum maximum mean size".split()

        if by == "minimum":
            print(structure_distances_grouped_by_kinases.groups)
            kinase_distances = structure_distances_grouped_by_kinases.min()
            kinase_distances = kinase_distances.reset_index().set_index(
                ["kinase1", "kinase2", "structure1", "structure2"]
            )
        elif by == "maximum":
            kinase_distances = structure_distances_grouped_by_kinases.max()
            kinase_distances = kinase_distances.reset_index().set_index(
                ["kinase1", "kinase2", "structure1", "structure2"]
            )
        elif by == "mean":
            kinase_distances = structure_distances_grouped_by_kinases.mean()
            kinase_distances = kinase_distances.reset_index().set_index(["kinase1", "kinase2"])
        elif by == "size":
            kinase_distances = structure_distances_grouped_by_kinases.size()
            kinase_distances.name = "n_structures"
            kinase_distances = kinase_distances.reset_index().set_index(["kinase1", "kinase2"])
        else:
            raise ValueError(f'Condition "by" unknown. Choose from: {", ".join(by_terms)}')

        return kinase_distances
