"""
kissim.encoding.features.subpockets 

Defines the subpockets feature.
"""

import logging

import numpy as np
import pandas as pd
from scipy.special import cbrt
from scipy.stats.stats import moment

from kissim.encoding.features import BaseFeature
from kissim.definitions import SUBPOCKETS

logger = logging.getLogger(__name__)


class SubpocketsFeature(BaseFeature):
    """
    Distances between all subpockets and all pocket residues.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
    _residue_ixs : list of int
        Residue indices.
    _distances : dict of (str: numpy.array)
        Distances between all subpockets and all pocket residues.
    _moments : dict of (str: list of float)
        Moments of distribution of distances between all subpockets and all pocket residues.
    """

    def __init__(self):
        self._residue_ids = None
        self._residue_ixs = None
        self._distances = None
        self._moments = None

    @classmethod
    def from_pocket(cls, pocket, subpockets=None):  # pylint: disable=W0221
        """
        Get feature from a pocket object.

        Parameters
        ----------
        pocket : kissim.io.PocketDataFrame
            Pocket object.
        subpockets : dict
            Dictionary with the following keys and values:
            "anchor_residue.klifs_ids" : list of int
                List of anchor residues (KLIFS residue IDs) whose centroid defines the subpocket
                center.
            "subpocket.name" : str
                Subpocket name.
            "subpocket.color" : str
                Subpocket color.
        """

        # If no subpockets are given, use global default subpockets
        subpockets = subpockets or SUBPOCKETS

        feature = cls()
        feature._residue_ids = pocket._residue_ids
        feature._residue_ixs = pocket._residue_ixs

        # Add subpockets
        pocket = feature._add_subpockets(pocket, subpockets)

        # Calculate distances
        feature._distances = feature._calculate_distances(pocket)

        # Calculate moments
        feature._moments = feature._calculate_moments()

        return feature

    @property
    def values(self):
        """
        Feature values for pocket residues.

        Returns
        -------
        dict of (str: list of float)
            Features for pocket residues.
        """
        return self._moments

    @property
    def details(self):
        """
        Feature details for pocket residues.

        Returns
        -------
        dict of pandas.DataFrame
            Feature details for pocket residues.
        """
        # Distances
        distances = pd.DataFrame(self._distances, index=self._residue_ixs)
        distances.insert(loc=0, column="residue.id", value=self._residue_ids)
        distances.index.name = "residue.ix"
        distances = distances.astype({"residue.id": "Int32"})

        # Moments
        moments = pd.DataFrame(self._moments, index=[1, 2, 3])
        moments.index.name = "moment"

        return {"distances": distances, "moments": moments}

    @staticmethod
    def _add_subpockets(pocket, subpockets):
        """
        Add subpockets to pocket object.

        Parameter
        ---------
        pocket : kissim.io.PocketDataFrame
            Pocket object.
        subpockets : dict
            Dictionary with the following keys and values (value lists must be of same length):
            "anchor_residue.klifs_id" : list of list of int
                List of anchor residues (KLIFS residue IDs) whose centroid defines the subpocket
                center.
            "subpocket.name" : list of str
                Subpocket name.
            "subpocket.color" : list of str
                Subpocket color.

        Returns
        -------
        kissim.io.PocketDataFrame
            Pocket object.
        """

        # Get residue PDB IDs for residue indices
        subpockets = pd.DataFrame(subpockets)
        subpockets["anchor_residue.ids"] = subpockets["anchor_residue.klifs_ids"].apply(
            lambda x: pocket.residues[pocket.residues["residue.ix"].isin(x)][
                "residue.id"
            ].to_list()
        )

        # Add subpockets
        for _, subpocket in subpockets.iterrows():
            pocket.add_subpocket(
                name=subpocket["subpocket.name"],
                anchor_residue_ixs=subpocket["anchor_residue.klifs_ids"],
                color=subpocket["subpocket.color"],
            )

        return pocket

    def _calculate_distances(self, pocket):
        """
        Calculate distances between all subpocket centers and all pocket residues (CA atoms).

        Parameters
        ----------
        pocket : kissim.io.PocketDataFrame
            Pocket object.

        Returns
        -------
        dict of (str: list of float)
            Distances to all pocket residues (values) from different subpockets (keys).
        """

        distances = {}

        # Distances to subpockets
        for _, subpocket in pocket.subpockets.iterrows():
            distances[subpocket["subpocket.name"]] = self._calculate_distances_to_center(
                pocket, subpocket["subpocket.center"]
            )

        # Distances to pocket center
        distances["center"] = self._calculate_distances_to_center(pocket, pocket.center)

        return distances

    def _calculate_distances_to_center(self, pocket, center):
        """
        Calculate distances between a subpocket or pocket center and all pocket residues (CA
        atoms).

        Parameters
        ----------
        pocket : kissim.io.PocketDataFrame
            Pocket object.
        center : numpy.array
            Subpocket or pocket center.

        Returns
        -------
        list of float
            Distances between a subpocket or pocket center and all pocket residues.
        """

        distances = []
        ca_atoms = pocket.ca_atoms

        for residue_id in pocket._residue_ids:
            if residue_id is None:
                distance = np.nan
            else:
                ca_atom_coord = ca_atoms[ca_atoms["residue.id"] == residue_id][
                    ["atom.x", "atom.y", "atom.z"]
                ].to_numpy()
                distance = np.linalg.norm(ca_atom_coord - center)
            distances.append(distance)
        return np.array(distances)

    def _calculate_moments(self):
        """
        Calculate moments of distributions of distances between all subpocket centers and all
        pocket residues (CA atoms).

        Parameters
        ----------
        pocket : kissim.io.PocketDataFrame
            Pocket object.

        Returns
        -------
        dict of (str: list of float)
            Distances to all pocket residues (values) from different subpockets (keys).
        """

        moments = {}
        for name, distances in self._distances.items():
            moment1, moment2, moment3 = self.calculate_first_second_third_moments(distances)
            moments[name] = np.array([moment1, moment2, moment3])
        return moments

    @staticmethod
    def calculate_first_second_third_moments(
        values,
    ):  # TODO Could be moved to something like utils
        """
        Get first, second, and third moment (mean, standard deviation, and skewness)
        for a distribution of values.
        Note: Moments are based only on non-NaN values.

        Parameters
        ----------
        values : list or numpy.array or pd.Series of (float or int)
            List of values.
        """

        values = np.array(values)

        if len(values) > 0:
            moment1 = np.nanmean(values)
            # Second and third moment: delta degrees of freedom = 0 (divisor N)
            moment2 = np.nanstd(values, ddof=0)
            moment3 = cbrt(moment(values, moment=3, nan_policy="omit"))
            return moment1, moment2, moment3
        else:
            return None, None, None
