"""
kissim.encoding.fingerprint_generator_base

Defines the base kissim fingerprint generator.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from kissim.encoding import FingerprintBase


logger = logging.getLogger(__name__)


class FingerprintGeneratorBase:
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
    """

    def __init__(self):
        self.structure_klifs_ids = None
        self.klifs_session = None
        self.data = None

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

    @property
    def physicochemical(self):
        """
        Get physicochemical feature vectors per feature type and pocket.

        Returns
        -------
        pandas.DataFrame
            Physicochemical feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("physicochemical")

    @property
    def distances(self):
        """
        Get distances feature vectors per feature type and pocket.

        Returns
        -------
        pandas.DataFrame
            Distances feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("distances")

    @property
    def moments(self):
        """
        Get moments feature vectors per feature type and pocket.

        Returns
        -------
        pandas.DataFrame
            Moments feature vectors per feature type (columns) and pocket (rows).
        """

        return self._feature_group("moments")

    @property
    def physicochemical_exploded(self):
        """
        Get physicochemical feature values per feature type and bit position.

        Returns
        -------
        pandas.DataFrame
            Physicochemical feature values per feature type (columns) and pocket / bit position
            (rows).
        """

        return self._feature_group_exploded("physicochemical")

    @property
    def distances_exploded(self):
        """
        Get distances feature values per feature type and bit position.

        Returns
        -------
        pandas.DataFrame
            Distances feature values per feature type (columns) and pocket / bit position (rows).
        """

        return self._feature_group_exploded("distances")

    @property
    def moments_exploded(self):
        """
        Get moments feature values per feature type and bit position.

        Returns
        -------
        pandas.DataFrame
            Moments feature values per feature type (columns) and pocket / bit position (rows).
        """

        return self._feature_group_exploded("moments")

    def _feature_group(self, feature_group):
        """
        For a given feature group, get feature vectors per feature type and pocket.

        Parameter
        ---------
        feature_group : str
            Feature group, i.e. "physicochemical", "distances", or "moments".

        Returns
        -------
        pandas.DataFrame
            Feature vectors per feature type (columns) and pocket (rows).
        """

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

    def _feature_group_exploded(self, feature_group):
        """
        For a given feature group, get moments feature values per feature type and bit position.

        Parameters
        ----------
        feature_group : str
            Feature group, i.e. "physicochemical", "distances", or "moments".

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
        features = self._feature_group(feature_group)
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

    @classmethod
    def from_json(cls, filepath):
        """
        Initialize a FingerprintGenerator object from a json file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to json file.
        """

        filepath = Path(filepath)
        with open(filepath, "r") as f:
            json_string = f.read()
        fingerprints_list = json.loads(json_string)

        data = {}
        for fingerprint_dict in fingerprints_list:
            fingerprint = FingerprintBase._from_dict(fingerprint_dict)
            data[fingerprint.structure_klifs_id] = fingerprint

        fingerprint_generator = cls()
        fingerprint_generator.data = data
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
