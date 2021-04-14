"""
kissim.comparison.feature_distances_generator

Defines the feature distances for multiple fingerprint pairs.
"""

import datetime
import json
import logging
from multiprocessing import Pool
from pathlib import Path

from itertools import combinations, repeat, chain

from kissim.encoding import FingerprintGenerator
from kissim.utils import set_n_cores
from . import FeatureDistances

logger = logging.getLogger(__name__)


class FeatureDistancesGenerator:
    """
    Generate feature distances for multiple fingerprint pairs, given a distance measure.
    Uses parallel computing of fingerprint pairs.

    Attributes
    ----------
    data : list of tuple of str: np.ndarray
        Feature distances and bit coverage (value) for each fingerprint pair.
    structure_kinase_ids : list of tuple
        Structure and kinase IDs for structures in dataset.
    """

    def __init__(self):
        self.data = None
        self.structure_kinase_ids = None

    @property
    def structure_ids(self):
        """
        Unique structure IDs associated with all fingerprints (sorted alphabetically).

        Returns
        -------
        list of str or int
            Structure IDs.
        """

        if self.data is not None:
            structure_ids = [i.structure_pair_ids for i in self.data]
            deduplicated_structure_ids = sorted(set(chain.from_iterable(structure_ids)))
            return deduplicated_structure_ids

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

        if self.data is not None:
            kinase_ids = [i.kinase_pair_ids for i in self.data]
            deduplicated_kinase_ids = sorted(set(chain.from_iterable(kinase_ids)))
            return deduplicated_kinase_ids

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
        logger.info(f"Number of input input fingerprints: {len(fingerprints_generator.data)}")

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
        feature_distances_generator.data = feature_distances_list
        feature_distances_generator.structure_kinase_ids = [
            (structure_id, fingerprint.kinase_name)
            for structure_id, fingerprint in fingerprints_generator.data.items()
        ]

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

    @classmethod
    def from_json(cls, filepath):
        """
        Initialize a FeatureDistancesGenerator object from a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.

        Returns
        -------
        kissim.comparison.FeatureDistancesGenerator
            Feature distances generator.
        """

        filepath = Path(filepath)
        with open(filepath, "r") as f:
            json_string = f.read()
        feature_distances_generator_dict = json.loads(json_string)

        data = {}
        for feature_distances_dict in feature_distances_generator_dict["data"]:
            feature_distances = FeatureDistances._from_dict(feature_distances_dict)
            data[feature_distances.structure_pair_ids] = feature_distances

        feature_distances_generator = cls()
        feature_distances_generator.data = data
        feature_distances_generator.structure_kinase_ids = feature_distances_generator_dict[
            "structure_kinase_ids"
        ]

        return feature_distances_generator

    def to_json(self, filepath):
        """
        Write FeatureDistancesGenerator class attributes to a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        feature_distances_generator_dict = self.__dict__.copy()

        # Format attribute data for JSON
        data = []
        for feature_distances in feature_distances_generator_dict["data"]:
            feature_distances_dict = feature_distances.__dict__.copy()
            feature_distances_dict["distances"] = feature_distances_dict["distances"].tolist()
            feature_distances_dict["bit_coverages"] = feature_distances_dict[
                "bit_coverages"
            ].tolist()
            data.append(feature_distances_dict)
        feature_distances_generator_dict["data"] = data

        json_string = json.dumps(feature_distances_generator_dict)
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.write(json_string)

    def by_structure_pair(self, structure_id1, structure_id2):
        """
        Get feature distances for fingerprint pair by their structure IDs, with details on
        feature types, feature names, and feature bit coverages.

        Parameters
        ----------
        structure_id1 : str
            Structure ID 1.
        structure_id2 : str
            Structure ID 2.

        Returns
        -------
        pandas.DataFrame
            Feature distances for fingerprint pair with details on feature types, features names,
            and feature bit coverages.
        """

        if self.data is not None:
            feature_distances = [
                i
                for i in self.data
                if (i.structure_pair_ids == (structure_id1, structure_id2))
                or (i.structure_pair_ids == (structure_id2, structure_id1))
            ]
            if len(feature_distances) != 1:
                raise ValueError(
                    f"{len(feature_distances)} entries for distance between "
                    f"{structure_id1} and {structure_id2} available, only 1 allowed."
                )
            else:
                feature_distances = feature_distances[0]
                return feature_distances.data

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
