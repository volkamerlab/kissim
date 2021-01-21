"""
kissim.encoding.features.sco 

Defines the side chain orientation feature.
"""

import logging

import pandas as pd
import numpy as np
from Bio.PDB import calc_angle

from kissim.encoding.features import BaseFeature

logger = logging.getLogger(__name__)


class SideChainOrientationFeature(BaseFeature):
    """
    Side chain orientation for each pocket residue.

    Attributes
    ----------
    name : str or int
        Name for structure encoding by this feature.
    _residue_ids : list of int
        Residue IDs.
    _residue_ixs : list of int
        Residue indices.
    _categories : list of float or None
        Pocket residues' side chain orientation categories.
    _vertex_angles : list of float or None
        Pocket residues' side chain orientation angles.
    _pocket_center : Bio.PDB.Vector.Vector
        Coordinates for the pocket's centroid.
    _ca_atoms : list of Bio.PDB.Vector.Vector or None
        Coordinates for the pocket residues' CA atoms.
    _sc_atoms : list of Bio.PDB.Vector.Vector or None
        Coordinates for the pocket residues' side chain representatives.

    Notes
    -----
    Side chain orientation of a residue is defined by the vertex angle formed by
    (i) the residue's CA atom,
    (ii) the residue's side chain centroid, and
    (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
    vertex.
    """

    def __init__(self):

        self.name = None
        self._residue_ids = None
        self._residue_ixs = None
        self._categories = None
        self._vertex_angles = None
        self._pocket_center = None
        self._ca_atoms = None
        self._sc_atoms = None

    @classmethod
    def from_pocket(cls, pocket):
        """
        Get side chain orientation for each pocket residue from a Biopython-based pocket object.

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython
            Biopython-based pocket object.

        Returns
        -------
        kissim.encoding.SideChainOrientationFeature
            Side chain orientation feature object.

        Notes
        -----
        Side chain orientation of a residue is defined by the vertex angle formed by
        (i) the residue's CA atom,
        (ii) the residue's side chain centroid, and
        (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
        vertex.
        """

        feature = cls()
        feature.name = pocket.name
        feature._residue_ids = pocket._residue_ids
        feature._residue_ixs = pocket._residue_ixs
        feature._pocket_center = pocket.center
        feature._ca_atoms = pocket.ca_atoms["ca.vector"].to_list()
        feature._sc_atoms = pocket.side_chain_representatives["sc.vector"].to_list()
        feature._vertex_angles = [
            feature._calculate_vertex_angle(sc_atom, ca_atom, feature._pocket_center)
            for ca_atom, sc_atom in zip(feature._ca_atoms, feature._sc_atoms)
        ]
        feature._categories = [
            feature._get_category(vertex_angle) for vertex_angle in feature._vertex_angles
        ]
        return feature

    @property
    def values(self):
        """
        Side chain orientation features for pocket residues.

        Returns
        -------
        list of float
            Side chain orientation features for pocket residues.
        """
        return self._categories

    @property
    def details(self):
        """
        Side chain orientation features for pocket residues (verbose).

        Returns
        -------
        pandas.DataFrame
            Side chain orientation features for pocket residues (rows) with the following columns:
            - "sco.category": Side chain orientation categories
            - "sco.angle": Side chain orientation angles
            - "ca.vector", "sc.vector", and "pocket_center.vector": Coordinates used for the angle calculation,
              i.e. the pocket centroid, pocket CA atoms, and pocket side chain representative.
        """

        features = pd.DataFrame(
            {
                "residue.id": self._residue_ids,
                "sco.category": self._categories,
                "sco.angle": self._vertex_angles,
                "ca.vector": self._ca_atoms,
                "sc.vector": self._sc_atoms,
            },
            index=self._residue_ixs,
        )
        features["pocket_center.vector"] = self._pocket_center
        features.index.name = "residue.ix"
        return features

    def _calculate_vertex_angle(self, vector1, vector2, vector3):
        """
        Calculate a vertex angle between three vectors (vertex = second vector).

        Parameters
        ----------
        vector1 : Bio.PDB.Vector.Vector or None
            Coordinates.
        vector2 : Bio.PDB.Vector.Vector or None
            Coordinates (defined as vertex of angle).
        vector2 : Bio.PDB.Vector.Vector or None
            Coordinates.

        Returns
        -------
        float or np.nan
            Vertex angle between the three points. None if any of the input vectors are None.
        """
        if all([vector1, vector2, vector2]):
            vertex_angle = np.degrees(calc_angle(vector1, vector2, vector3))
            vertex_angles = vertex_angle.round(2)
            return vertex_angle
        else:
            return np.nan

    def _get_category(self, vertex_angle):
        """
        Transform a given vertex angle into a category value, which defines the side chain
        orientation towards the pocket:
        - inwards (category 0.0)
        - intermediate (category 1.0)
        - outwards (category 2.0)

        Parameters
        ----------
        vertex_angle : float or None
            Vertex angle between a residue's CA atom (vertex), side chain representative and pocket
            centroid. Ranges between 0.0 and 180.0.

        Returns
        -------
        float or None
            Category for side chain orientation towards pocket.
            None if any of the input vectors are None.
        """

        if np.isnan(vertex_angle):
            return np.nan
        elif 0.0 <= vertex_angle <= 45.0:  # Inwards
            return 0.0
        elif 45.0 < vertex_angle <= 90.0:  # Intermediate
            return 1.0
        elif 90.0 < vertex_angle <= 180.0:  # Outwards
            return 2.0
        else:
            raise ValueError(
                f"Molecule {self.name}: Unknown vertex angle {vertex_angle}. "
                f"Only values between 0.0 and 180.0 allowed."
            )
