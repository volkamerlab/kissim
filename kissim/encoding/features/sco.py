"""
kissim.encoding.features.sco TODO
"""

import logging

import pandas as pd
import numpy as np
from Bio.PDB import calc_angle

logger = logging.getLogger(__name__)


class SideChainOrientationFeature:
    """
    Side chain orientation for each residue in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues.
    Side chain orientation of a residue is defined by the vertex angle formed by
    (i) the residue's CA atom,
    (ii) the residue's side chain centroid, and
    (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
    vertex.

    Attributes
    ----------
    features : pandas.DataFrame
        1 feature, i.e. side chain orientation, (column) for 85 residues (rows).
    features_verbose : pandas.DataFrame
        Feature, Ca, Cb, and centroid vectors as well as metadata information (columns)
        for 85 residues (row).
    """

    def __init__(self):

        self.residue_ids = None
        self._categories = None
        self._vertex_angles = None
        self._centroid = None
        self._ca_atoms = None
        self._sc_atoms = None

    @classmethod
    def from_pocket_biopython(cls, pocket):
        """
        Get side chain orientation for each residue in a molecule (pocket).
        Side chain orientation of a residue is defined by the vertex angle formed by
        (i) the residue's CA atom,
        (ii) the residue's side chain centroid, and
        (iii) the pocket centroid (calculated based on its CA atoms), whereby the CA atom forms the
        vertex.

        Parameters
        ----------
        pocket : kissim.io.biopython.pocket.PocketBiopython
            TODO
        """

        feature = cls()
        feature.residue_ids = pocket.residue_ids

        centroid = feature._get_centroid(pocket)
        ca_atoms = pocket.ca_atoms["ca.vector"].to_list()
        sc_atoms = [
            feature._get_side_chain_representative(pocket, residue_id)
            for residue_id in feature.residue_ids
        ]

        vertex_angles = [
            feature._calculate_vertex_angle(sc_atom, ca_atom, centroid)
            if all([sc_atom, ca_atom, centroid])
            else None
            for ca_atom, sc_atom in zip(ca_atoms, sc_atoms)
        ]
        categories = [
            feature._get_category(vertex_angle) if vertex_angle else None
            for vertex_angle in vertex_angles
        ]

        feature._categories = categories
        feature._vertex_angles = vertex_angles
        feature._centroid = centroid
        feature._ca_atoms = ca_atoms
        feature._sc_atoms = sc_atoms

        return feature

    @property
    def features(self):
        """TODO"""

        features = pd.DataFrame(self._categories, columns=["sco"], index=self.residue_ids)

        return features

    @property
    def features_verbose(self):
        """TODO"""

        features = pd.DataFrame(
            {
                "sco.category": self._categories,
                "sco.angle": self._vertex_angles,
                "ca.vector": self._ca_atoms,
                "sc.vector": self._sc_atoms,
            },
            index=self.residue_ids,
        )
        features["centroid"] = self._centroid

        return features

    def _get_side_chain_representative(self, pocket, residue_id):
        """TODO"""

        atom = pocket._side_chain_representative(residue_id)
        if atom:
            vector = atom.get_vector()
            return vector
        else:
            vector = pocket._pcb_atom(residue_id)
            return vector

        return vector

    def _get_centroid(self, pocket):
        """TODO"""

        vector = pocket.centroid
        return vector

    def _calculate_vertex_angle(self, vector1, vector2, vector3):
        """
        Calculate a vertex angle between three vectors (vertex = second vector).
        """

        vertex_angle = np.degrees(calc_angle(vector1, vector2, vector3))
        vertex_angles = vertex_angle.round(2)
        return vertex_angle

    def _get_category(self, vertex_angle):
        """
        Transform a given vertex angle into a category value, which defines the side chain
        orientation towards the pocket:
        - inwards (category 0.0)
        - intermediate (category 1.0)
        - outwards (category 2.0)

        Parameters
        ----------
        vertex_angle : float
            Vertex angle between a residue's CA atom (vertex), side chain representative and pocket
            centroid. Ranges between 0.0 and 180.0.

        Returns
        -------
        float
            Category for side chain orientation towards pocket.
        """

        if 0.0 <= vertex_angle <= 45.0:  # Inwards
            return 0.0
        elif 45.0 < vertex_angle <= 90.0:  # Intermediate
            return 1.0
        elif 90.0 < vertex_angle <= 180.0:  # Outwards
            return 2.0
        elif np.isnan(vertex_angle):
            return np.nan
        else:
            raise ValueError(
                f"Molecule {self.molecule_code}: Unknown vertex angle {vertex_angle}. "
                f"Only values between 0.0 and 180.0 allowed."
            )
