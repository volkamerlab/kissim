"""
kissim.comparison.fingerprint_distance_generator

Defines the pairwise fingerprint distances for a set of fingerprints.
"""

import datetime
import logging

from tqdm.auto import tqdm

from kissim.comparison import BaseGenerator, FingerprintDistance, FeatureDistancesGenerator
from kissim.comparison import matrix
from kissim.comparison.utils import format_weights

logger = logging.getLogger(__name__)


class FingerprintDistanceGenerator(BaseGenerator):
    """
    Generate fingerprint distances for multiple fingerprint pairs based on their feature distances,
    given a feature weighting scheme.

    Attributes
    ----------
    data : pandas.DataFrame
        Fingerprint distance and bit coverage for each structure pair (kinase pair).
    structure_kinase_ids : list of list
        Structure and kinase IDs for structures in dataset.
    """

    def __init__(self, *args, **kwargs):
        self.data = None
        self.structure_kinase_ids = None

    def __eq__(self, other):
        if isinstance(other, FingerprintDistanceGenerator):
            return (
                self.data.equals(other.data)
                and self.structure_kinase_ids == other.structure_kinase_ids
            )

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
            (ii) By feature (list of 15 floats):
                Features to be set in the following order: size, hbd, hba, charge, aromatic,
                aliphatic, sco, exposure, distance_to_centroid, distance_to_hinge_region,
                distance_to_dfg_region, distance_to_front_pocket, moment1, moment2, and moment3.
                All floats must sum up to 1.0.

        Returns
        -------
        kissim.comparison.FingerprintDistanceGenerator
            Fingerprint distance generator.
        """

        logger.info("GENERATE FINGERPRINT DISTANCES")
        # logger.info(f"Number of input feature distances: {len(feature_distances_generator.data)}")

        start_time = datetime.datetime.now()
        logger.info(f"Fingerprint distance generation started at: {start_time}")

        # Format input feature weights
        feature_weights = format_weights(feature_weights)
        logger.info(f"Feature weights: {feature_weights}")

        # Weighted sum of pairwise feature distances and bit coverages
        fingerprint_distance = FingerprintDistance()
        distances = [
            fingerprint_distance._distance(distances, feature_weights)
            for distances in tqdm(
                feature_distances_generator.distances,
                desc="Calculate pairwise fingerprint distance",
            )
        ]
        bit_coverages = [
            fingerprint_distance._bit_coverage(bit_coverages, feature_weights)
            for bit_coverages in tqdm(
                feature_distances_generator.bit_coverages,
                desc="Calculate pairwise fingerprint coverage",
            )
        ]

        # Set class attributes

        fingerprint_distance_generator = cls()
        fingerprint_distance_generator.data = feature_distances_generator.data[
            ["structure.1", "structure.2", "kinase.1", "kinase.2"]
        ].copy()
        fingerprint_distance_generator.data["distance"] = distances
        fingerprint_distance_generator.data["bit_coverage"] = bit_coverages
        fingerprint_distance_generator.structure_kinase_ids = (
            feature_distances_generator.structure_kinase_ids
        )

        logger.info(
            f"Number of output fingerprint distances: {len(fingerprint_distance_generator.data)}"
        )

        end_time = datetime.datetime.now()
        logger.info(f"Runtime: {end_time - start_time}")

        return fingerprint_distance_generator

    @classmethod
    def from_structure_klifs_ids(
        cls, structure_klifs_ids, klifs_session=None, feature_weights=None, n_cores=1
    ):
        """
        Calculate fingerprint distances for all possible structure pairs.

        Parameters
        ----------
        structure_klifs_id : int
            Input structure KLIFS ID (output fingerprints may contain less IDs because some
            structures could not be encoded).
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.
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
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        kissim.comparison.FingerprintDistancesGenerator
            Fingerprint distance generator.
        """

        feature_distances_generator = FeatureDistancesGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores
        )
        fingerprint_distance_generator = cls.from_feature_distances_generator(
            feature_distances_generator, feature_weights
        )
        return fingerprint_distance_generator

    @classmethod
    def from_fingerprint_generator(cls, fingerprint_generator, feature_weights=None, n_cores=1):
        """
        Calculate fingerprint distances for all possible structure pairs.

        Parameters
        ----------
        fingerprint_generator : kissim.encoding.FingerprintGenerator
            Fingerprints.
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
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        kissim.comparison.FingerprintDistancesGenerator
            Fingerprint distance generator.
        """

        feature_distances_generator = FeatureDistancesGenerator.from_fingerprint_generator(
            fingerprint_generator, n_cores
        )
        fingerprint_distance_generator = cls.from_feature_distances_generator(
            feature_distances_generator, feature_weights
        )
        return fingerprint_distance_generator

    def structure_distance_matrix(self, coverage_min=0.0):
        """
        Get fingerprint distances for all structure pairs in the form of a matrix (DataFrame).

        Parameters
        ----------
        fill : bool
            Fill or fill not (default) lower triangle of distance matrix.
        coverage_min : float
            Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
            coverage restrictions).

        Returns
        -------
        pandas.DataFrame
            Structure distance matrix.
        """

        return matrix.structure_distance_matrix(self.data, coverage_min)

    def kinase_distance_matrix(self, by="minimum", fill_diagonal=True, coverage_min=0.0):
        """
        Extract per kinase pair one distance value from the set of structure pair distance values
        and return these  fingerprint distances for all kinase pairs in the form of a matrix
        (DataFrame).

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.
        fill_diagonal : bool
            Fill diagonal with 0 (same kinase has distance of 0) by default. If `False`, diagonal
            will be a experimental values calculated based on the structure pairs per kinase pair.
            Is by default set to False, if `by="size"`.
        coverage_min : float
            Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
            coverage restrictions).

        Returns
        -------
        pandas.DataFrame
            Kinase distance matrix.
        """

        return matrix.kinase_distance_matrix(self.data, by, fill_diagonal, coverage_min)

    def kinase_distances(self, by="minimum", coverage_min=0.0):
        """
        Extract per kinase pair one distance value from the set of structure pair distance values.

        Parameters
        ----------
        by : str
            Condition on which the distance value per kinase pair is extracted from the set of
            distances values per structure pair. Default: Minimum distance value.
        coverage_min : float
            Returns only pairs with a user-defined minimum coverage (defaults to 0.0, i.e. no
            coverage restrictions).

        Returns
        -------
        pandas.DataFrame
            Fingerprint distance and coverage for kinase pairs.
        """

        return matrix.kinase_distances(self.data, by, coverage_min)
