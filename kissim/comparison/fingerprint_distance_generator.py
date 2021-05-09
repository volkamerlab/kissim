"""
kissim.comparison.fingerprint_distance_generator

Defines the pairwise fingerprint distances for a set of fingerprints.
"""

import datetime
import logging

from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from kissim.comparison import BaseGenerator, FingerprintDistance, FeatureDistancesGenerator
from kissim.comparison.utils import format_weights

logger = logging.getLogger(__name__)


class FingerprintDistanceGenerator(BaseGenerator):
    """
    Generate fingerprint distances for multiple fingerprint pairs based on their feature distances,
    given a feature weighting scheme.

    Attributes
    ----------
    data : pandas.DataFrame
        Fingerprint distance and bit coverag for each structure pair (kinase pair).
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
        pairs_upper = self.data[["structure.1", "structure.2", "distance"]]
        # Data for lower half of the matrix
        pairs_lower = pairs_upper.rename(
            columns={"structure.1": "structure.2", "structure.2": "structure.1"}
        )

        # Concatenate upper and lower matrix data
        pairs = pd.concat([pairs_upper, pairs_lower]).sort_values(["structure.1", "structure.2"])
        # Convert to matrix
        matrix = pairs.pivot(columns="structure.2", index="structure.1", values="distance")
        # Matrix diagonal is NaN > set to 0.0
        matrix = matrix.fillna(0.0)

        return matrix

    def kinase_distance_matrix(self, by="minimum", fill_diagonal=True):
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

        Returns
        -------
        pandas.DataFrame
            Kinase distance matrix.
        """

        if by == "size":
            fill_diagonal = False

        # Data for upper half of the matrix
        pairs_upper = self.kinase_distances(by).reset_index()[["kinase.1", "kinase.2", "distance"]]
        # Data for lower half of the matrix
        pairs_lower = pairs_upper.rename(columns={"kinase.1": "kinase.2", "kinase.2": "kinase.1"})

        # Concatenate upper and lower matrix data
        pairs = (
            pd.concat([pairs_upper, pairs_lower])
            .sort_values(["kinase.1", "kinase.2"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Convert to matrix
        matrix = pairs.pivot(columns="kinase.2", index="kinase.1", values="distance")

        if fill_diagonal:
            np.fill_diagonal(matrix.values, 0)

        # If matrix contains number of structure pairs: NaN > 0, cast to int
        if by == "size":
            matrix = matrix.fillna(0)
            matrix = matrix.astype("int")

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

        data = self.data
        # Group by kinase names
        structure_distances_grouped_by_kinases = data.groupby(
            by=["kinase.1", "kinase.2"], sort=False
        )

        # Get distance values per kinase pair based on given condition
        # Note: For min/max we'd like to know which structure pairs were selected!
        by_terms = "minimum maximum mean median size std".split()

        if by == "minimum":
            kinase_distances = data.iloc[
                structure_distances_grouped_by_kinases["distance"].idxmin()
            ].set_index(["kinase.1", "kinase.2"])
        elif by == "maximum":
            kinase_distances = data.iloc[
                structure_distances_grouped_by_kinases["distance"].idxmax()
            ].set_index(["kinase.1", "kinase.2"])
        elif by == "mean":
            kinase_distances = structure_distances_grouped_by_kinases.mean()[["distance"]]
        elif by == "median":
            kinase_distances = structure_distances_grouped_by_kinases.median()[["distance"]]
        elif by == "size":
            kinase_distances = structure_distances_grouped_by_kinases.size().to_frame("distance")
        elif by == "std":
            kinase_distances = structure_distances_grouped_by_kinases.std()[["distance"]]
            kinase_distances = round(kinase_distances, 3)
        else:
            raise ValueError(f'Condition "by" unknown. Choose from: {", ".join(by_terms)}')

        return kinase_distances
