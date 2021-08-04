"""
kissim.encoding.fingerprint

Defines sequencial and parallel processing of fingerprints from local or remote structures.
"""

import datetime
from itertools import repeat
import json
import logging
from pathlib import Path

from multiprocessing import Pool
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.encoding import Fingerprint, FingerprintNormalized
from kissim.utils import set_n_cores

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """
    Generate fingerprints for multiple structures.

    Attributes
    ----------
    structure_klifs_id : int
        Structure KLIFS ID.
    klifs_session : opencadd.databases.klifs.session.Session
        Local or remote KLIFS session.
    data : dict of int: kissim.encoding.Fingerprint
        Fingerprints for input structures (by KLIFS ID).
    data_normalized : dict of int: kissim.encoding.Fingerprint
        Normalized fingerprints for input structures (by KLIFS ID).
    """

    def __init__(self):

        self.structure_klifs_ids = None
        self.klifs_session = None
        self.data = None
        self.data_normalized = None

    @classmethod
    def from_structure_klifs_ids(cls, structure_klifs_ids, klifs_session=None, n_cores=1):
        """
        Calculate fingerprints for one or more KLIFS structures (by structure KLIFS IDs).

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
        kissim.encoding.fingerprint_generator
            Fingerprint generator object containing fingerprints.
        """

        logger.info("GENERATE FINGERPRINTS")
        logger.info(f"Number of input structures: {len(structure_klifs_ids)}")

        start_time = datetime.datetime.now()
        logger.info(f"Fingerprint generation started at: {start_time}")

        # Set up KLIFS session if needed
        if klifs_session is None:
            klifs_session = setup_remote()

        # Set number of cores to be used
        n_cores = set_n_cores(n_cores)

        # Initialize FingerprintGenerator object
        fingerprint_generator = cls()
        fingerprint_generator.structure_klifs_ids = structure_klifs_ids
        fingerprint_generator.klifs_session = klifs_session
        fingerprints_list = fingerprint_generator._get_fingerprint_list(n_cores)
        fingerprint_generator.data = {
            i.structure_klifs_id: i
            for i in fingerprints_list
            if i is not None  # Removes emtpy fingerprints
        }
        fingerprint_generator.data_normalized = fingerprint_generator._normalize_fingerprints()

        logger.info(f"Number of output fingerprints: {len(fingerprint_generator.data)}")

        end_time = datetime.datetime.now()
        logger.info(f"Runtime: {end_time - start_time}")

        return fingerprint_generator

    @classmethod
    def from_json(cls, filepath, normalize=False):
        """
        Initialize a FingerprintGenerator object from a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        normalized : bool
            Add normalization (default: False). This will store the unnormalized features alongside
            the normalized features.
        """

        filepath = Path(filepath)
        with open(filepath, "r") as f:
            json_string = f.read()
        fingerprints_list = json.loads(json_string)

        data = {}
        for fingerprint_dict in fingerprints_list:
            fingerprint = Fingerprint._from_dict(fingerprint_dict)
            data[fingerprint.structure_klifs_id] = fingerprint

        fingerprint_generator = cls()
        fingerprint_generator.data = data
        if normalize:
            fingerprint_generator.data_normalized = fingerprint_generator._normalize_fingerprints()
        fingerprint_generator.structure_klifs_ids = list(fingerprint_generator.data.keys())

        return fingerprint_generator

    def to_json(self, filepath):
        """
        Write FingerprintGenerator class attributes to a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        fingerprint_list = [
            fingerprint.__dict__ for structure_klifs_id, fingerprint in self.data.items()
        ]
        json_string = json.dumps(fingerprint_list)
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.write(json_string)

    @property
    def subpocket_centers(self):
        """
        Subpocket center coordinates for all structures.

        Returns
        -------
        pandas.DataFrame
            All subpockets (columns, level 0) coordinates x, y, z (columns, level 1) for all
            structures (rows).
        """

        coordinates = []
        for structure_klifs_id, fingerprint in self.data.items():
            coordinates_series = fingerprint.subpocket_centers.transpose().stack()
            coordinates_series.name = structure_klifs_id
            coordinates.append(coordinates_series)
        coordinates = pd.DataFrame(coordinates)
        coordinates.columns.names = ["subpocket", "dimension"]

        return coordinates

    def physicochemical(self, normalized=False):
        """
        Get physicochemical feature vectors per feature type and pocket.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Physicochemical feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("physicochemical", normalized)

    def distances(self, normalized=False):
        """
        Get distances feature vectors per feature type and pocket.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Distances feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("distances", normalized)

    def moments(self, normalized=False):
        """
        Get moments feature vectors per feature type and pocket.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Moments feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("moments", normalized)

    def physicochemical_exploded(self, normalized=False):
        """
        Get physicochemical feature values per feature type and bit position.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Physicochemical feature values per feature type (columns) and pocket / bit position
            (rows).
        """

        return self._feature_group_exploded("physicochemical", normalized)

    def distances_exploded(self, normalized=False):
        """
        Get distances feature values per feature type and bit position.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Distances feature values per feature type (columns) and pocket / bit position (rows).
        """

        return self._feature_group_exploded("distances", normalized)

    def moments_exploded(self, normalized=False):
        """
        Get moments feature values per feature type and bit position.

        Parameters
        ----------
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Moments feature values per feature type (columns) and pocket / bit position (rows).
        """

        return self._feature_group_exploded("moments", normalized)

    def _feature_group(self, feature_group, normalized=False):
        """
        For a given feature group, get feature vectors per feature type and pocket.

        Parameter
        ---------
        feature_group : str
            Feature group, i.e. "physicochemical", "distances", or "moments".
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Feature vectors per feature type (columns) and pocket (rows).
        """

        if normalized:
            fingerprints = self.data_normalized
        else:
            fingerprints = self.data

        if fingerprints is not None:
            features = {
                structure_klifs_id: (
                    fingerprint.values_dict[feature_group]
                    if feature_group == "physicochemical"
                    else fingerprint.values_dict["spatial"][feature_group]
                )
                for structure_klifs_id, fingerprint in fingerprints.items()
            }
            features = pd.DataFrame(features).transpose()
        else:
            features = None

        return features

    def _feature_group_exploded(self, feature_group, normalized=False):
        """
        For a given feature group, get moments feature values per feature type and bit position.

        Parameters
        ----------
        feature_group : str
            Feature group, i.e. "physicochemical", "distances", or "moments".
        normalized : bool
            Unnormalized (default) or normalized features.

        Returns
        -------
        pandas.DataFrame
            Feature values per feature type (columns) and pocket / bit position (rows).
        """

        index_level1 = "structure_klifs_id"
        if feature_group == "moments":
            index_level2 = "moment"
        else:
            index_level2 = "residue_ix"
        features = self._feature_group(feature_group, normalized)
        features_exploded = features.apply(lambda x: x.explode()).astype(float)
        features_exploded.index.name = index_level1
        multi_index = (
            features_exploded.groupby(index_level1, sort=False, dropna=False)
            .size()
            .apply(lambda x: range(1, x + 1))
            .explode()
        )
        multi_index = pd.MultiIndex.from_tuples(
            list(multi_index.items()), names=[index_level1, index_level2]
        )
        features_exploded.index = multi_index
        return features_exploded

    def _get_fingerprint_list(self, n_cores):
        """
        Generate fingerprints.

        Parameters
        ----------
        n_cores : int
            Number of cores.

        Returns
        -------
        list of kissim.encoding.fingerprint
            List of fingerprints
        """

        pool = Pool(processes=n_cores)
        fingerprints_list = pool.starmap(
            self._get_fingerprint, zip(self.structure_klifs_ids, repeat(self.klifs_session))
        )
        pool.close()
        pool.join()

        return fingerprints_list

    def _get_fingerprint(self, structure_klifs_id, klifs_session):
        """
        Generate a fingerprint.

        Parameters
        ----------
        structure_klifs_id : int
            Structure KLIFS ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        kissim.encoding.fingerprint
            Fingerprint.
        """

        logger.info(f"{structure_klifs_id}: Generate fingerprint...")
        fingerprint = Fingerprint.from_structure_klifs_id(structure_klifs_id, klifs_session)
        return fingerprint

    def _normalize_fingerprints(self):
        """
        Normalize fingerprints in fingerprint generator.

        Returns
        -------
        dict
            Normalized fingerprints (values) by fingerprint ID (keys).
        """
        return {
            key: FingerprintNormalized.from_fingerprint(value) for key, value in self.data.items()
        }
