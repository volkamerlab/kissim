"""
kissim.comparison.feature_distances_generator

Defines the feature distances for multiple fingerprint pairs.
"""

import datetime
import logging
from multiprocessing import Pool
from itertools import combinations, repeat

import pandas as pd

from kissim.utils import set_n_cores
from kissim.encoding import FingerprintGenerator
from kissim.comparison import BaseGenerator, FeatureDistances

logger = logging.getLogger(__name__)


class FeatureDistancesGenerator(BaseGenerator):
    """
    Generate feature distances for multiple fingerprint pairs, given a distance measure.
    Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    data : pandas.DataFrame
        Feature distances and bit coverages for each structure pair (kinase pair).
    structure_kinase_ids : list of list
        Structure and kinase IDs for structures in dataset.
    """

    def __init__(self, **kwargs):
        self.data = None
        self.structure_kinase_ids = None

    def __eq__(self, other):

        if isinstance(other, FeatureDistancesGenerator):
            return (
                self.data.equals(other.data)
                and self.structure_kinase_ids == other.structure_kinase_ids
            )

    @classmethod
    def from_fingerprint_generator(cls, fingerprints_generator, n_cores=1):
        """
        Calculate feature distances for all possible structure pairs.

        Parameters
        ----------
        fingerprints_generator : kissim.encoding.FingerprintsGenerator
            Multiple fingerprints.
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        kissim.comparison.FeatureDistancesGenerator
            Feature distances generator.
        """

        logger.info("GENERATE FEATURE DISTANCES")
        logger.info(f"Number of input fingerprints: {len(fingerprints_generator.data)}")

        start_time = datetime.datetime.now()
        logger.info(f"Feature distances generation started at: {start_time}")

        # Set number of cores to be used
        n_cores = set_n_cores(n_cores)

        # Initialize FeatureDistancesGenerator object
        feature_distances_generator = cls()
        feature_distances_list = feature_distances_generator._get_feature_distances_from_list(
            feature_distances_generator._get_feature_distances,
            fingerprints_generator.data,
            n_cores,
        )
        feature_distances_generator.data = (
            feature_distances_generator._feature_distances_list_to_df(feature_distances_list)
        )
        feature_distances_generator.structure_kinase_ids = (
            feature_distances_generator._structure_kinase_ids
        )

        logger.info(f"Number of ouput feature distances: {len(feature_distances_generator.data)}")

        end_time = datetime.datetime.now()
        logger.info(f"Runtime: {end_time - start_time}")

        return feature_distances_generator

    @classmethod
    def from_structure_klifs_ids(cls, structure_klifs_ids, klifs_session=None, n_cores=1):
        """
        Calculate feature distances for all possible structure pairs.

        Parameters
        ----------
        structure_klifs_id : int
            Input structure KLIFS ID (output fingerprints may contain less IDs because some
            structures could not be encoded).
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        kissim.comparison.FeatureDistancesGenerator
            Feature distances generator.
        """

        fingerprint_generator = FingerprintGenerator.from_structure_klifs_ids(
            structure_klifs_ids, klifs_session, n_cores
        )
        feature_distances_generator = cls.from_fingerprint_generator(
            fingerprint_generator, n_cores
        )
        return feature_distances_generator

    @staticmethod
    def _fingerprint_pairs(fingerprints):
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

        pairs = [(i, j) for i, j in combinations(fingerprints.keys(), 2)]
        return pairs

    def _get_feature_distances_from_list(self, _get_feature_distances, fingerprints, n_cores):
        """
        Get feature distances for multiple fingerprint pairs.
        Uses parallel computing.

        Parameters
        ----------
        _get_feature_distances : method
            Method calculating feature distances for one fingerprint pair.
        fingerprints : dict of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.
        n_cores : int or None
            Number of cores to be used for fingerprint generation as defined by the user.

        Returns
        -------
        list of kissim.similarity.FeatureDistances
            List of distances and bit coverages between two fingerprints for each of their
            features.
        """

        pool = Pool(processes=n_cores)
        pairs = self._fingerprint_pairs(fingerprints)
        feature_distances_list = pool.starmap(
            _get_feature_distances, zip(pairs, repeat(fingerprints))
        )
        pool.close()
        pool.join()

        return feature_distances_list

    @staticmethod
    def _get_feature_distances(pair, fingerprints):
        """
        Calculate the feature distances for one fingerprint pair.

        Parameters
        ----------
        fingerprints : dict of tuple of str: kissim.encoding.Fingerprint
            Dictionary of fingerprints: Keys are molecule codes and values are fingerprint data.

        Returns
        -------
        kissim.similarity.FeatureDistances
            Distances and bit coverages between two fingerprints for each of their features.
        """

        fingerprint1 = fingerprints[pair[0]]
        fingerprint2 = fingerprints[pair[1]]

        feature_distances = FeatureDistances.from_fingerprints(fingerprint1, fingerprint2)

        return feature_distances

    @staticmethod
    def _feature_distances_list_to_df(feature_distances_list):

        structure_pair_ids_list = []
        kinase_pair_ids_list = []
        distances_list = []
        bit_coverages_list = []

        for feature_distances in feature_distances_list:
            structure_pair_ids_list.append(feature_distances.structure_pair_ids)
            kinase_pair_ids_list.append(feature_distances.kinase_pair_ids)
            distances_list.append(feature_distances.distances)
            bit_coverages_list.append(feature_distances.bit_coverages)

        return pd.concat(
            [
                pd.DataFrame(structure_pair_ids_list, columns=["structure.1", "structure.2"]),
                pd.DataFrame(kinase_pair_ids_list, columns=["kinase.1", "kinase.2"]),
                pd.DataFrame(
                    distances_list,
                    columns=[f"distance.{i}" for i in range(1, len(distances_list[0]) + 1)],
                ),
                pd.DataFrame(
                    bit_coverages_list,
                    columns=[
                        f"bit_coverage.{i}" for i in range(1, len(bit_coverages_list[0]) + 1)
                    ],
                ),
            ],
            axis=1,
        )
