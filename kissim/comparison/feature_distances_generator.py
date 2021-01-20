"""
kissim.comparison.feature_distances_generator

Defines the feature distances for multiple fingerprint pairs.
"""

import datetime
import logging
from multiprocessing import cpu_count, Pool

from itertools import combinations, repeat, chain
import pandas as pd

from kissim.encoding.schema import FEATURE_NAMES
from . import FeatureDistances

logger = logging.getLogger(__name__)


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
        Calculate feature distances for all possible fingerprint pair combinations, given a
        distance measure.

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
        Get feature distances for fingerprint pair by their molecule codes, with details on
        feature types, feature names, and feature bit coverages.

        Parameters
        ----------
        molecule_code1 : str
            Molecule code 1.
        molecule_code2 : str
            Molecule code 2.

        Returns
        -------
        pandas.DataFrame
            Feature distances for fingerprint pair with details on feature types, features names,
            and feature bit coverages.
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
            List of distances and bit coverages between two fingerprints for each of their
            features.
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
            Dictionary of non-empty fingerprints: Keys are molecule codes and values are
            fingerprint data.
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
