"""
kissim.io.biopython

Defines a Biopython-based pocket class.
"""

import pandas as pd
from Bio.PDB import HSExposure, Vector, Entity


from ..definitions import SIDE_CHAIN_REPRESENTATIVE
from .core import Pocket


class PocketBiopython(Pocket):
    """
    Class defining the base for structural objects for this package. TODO
    """

    def __init__(self):

        self._data_complex = None
        self._pocket_residue_ids = None
        self._hse_ca = None
        self._hse_cb = None

    @classmethod
    def from_local(cls, local, structure_id):
        """TODO"""

        return cls._from_backend(local, structure_id)

    @classmethod
    def from_remote(cls, remote, structure_id):
        """TODO"""

        return cls._from_backend(remote, structure_id)

    @classmethod
    def _from_backend(cls, backend, structure_id):
        """TODO"""

        pocket = cls()
        pocket._data_complex = pocket._get_biopython(backend, structure_id)
        pocket._pocket_residue_ids = pocket._get_pocket_residue_ids(backend, structure_id)
        # Cast residue IDs str > int (necessary for Biopython where they are int)
        pocket._pocket_residue_ids = [int(i) for i in pocket._pocket_residue_ids]
        pocket._hse_ca = HSExposure.HSExposureCA(pocket._data_complex)
        pocket._hse_cb = HSExposure.HSExposureCB(pocket._data_complex)
        return pocket

    @property
    def residue_ids(self):
        """TODO"""
        return self._pocket_residue_ids

    @property
    def centroid(self):
        """TODO"""

        ca_atoms = self.ca_atoms
        ca_atom_vectors = ca_atoms["ca.atom"].to_list()
        ca_atom_vectors = [i for i in ca_atom_vectors if i != None]
        centroid = self.center_of_mass(ca_atom_vectors, geometric=False)
        centroid = Vector(centroid)

        return centroid

    @property
    def ca_atoms(self):
        """TODO"""

        ca_atoms = []
        for residue_id in self.residue_ids:
            ca_atom = self._ca_atom(residue_id)
            ca_atoms.append([residue_id, ca_atom])
        ca_atoms = pd.DataFrame(ca_atoms, columns=["residue.id", "ca.atom"])

        # Add vectors
        ca_atom_vectors = []
        for ca_atom in ca_atoms["ca.atom"]:
            try:
                ca_atom_vectors.append(ca_atom.get_vector())
            except AttributeError:
                ca_atom_vectors.append(None)
        ca_atoms["ca.vector"] = ca_atom_vectors

        return ca_atoms

    @property
    def pcb_atoms(self):
        """TODO"""

        pcb_atoms = []
        for residue_id in self.residue_ids:
            pcb_atom = self._pcb_atom(residue_id)
            pcb_atoms.append([residue_id, pcb_atom])
        pcb_atoms = pd.DataFrame(pcb_atoms, columns=["residue.id", "pcb.vector"])

        return pcb_atoms

    @property
    def side_chain_representatives(self):
        """TODO"""

        sc_atoms = []
        for residue_id in self.residue_ids:
            sc_atom = self._side_chain_representative(residue_id)
            sc_atoms.append([residue_id, sc_atom])
        sc_atoms = pd.DataFrame(sc_atoms, columns=["residue.id", "sc.atom"])

        # Add vectors
        sc_atom_vectors = []
        for sc_atom in sc_atoms["sc.atom"]:
            try:
                sc_atom_vectors.append(sc_atom.get_vector())
            except AttributeError:
                sc_atom_vectors.append(None)
        sc_atoms["sc.vector"] = sc_atom_vectors

        return sc_atoms

    def _ca_atom(self, residue_id):
        """TODO"""
        residue = self._residue_from_residue_id(residue_id)
        try:
            return residue["CA"]
        except KeyError:
            return None

    def _pcb_atom(self, residue_id):
        """TODO"""

        # Get biopython residue object
        residue = self._residue_from_residue_id(residue_id)

        # Get pCB atom
        if residue.get_resname() == "GLY":
            pcb = self._pcb_atom_from_gly(residue)
        else:
            pcb = self._pcb_atom_from_non_gly(residue)

        if pcb:
            # Center pseudo-CB vector at CA atom to get pseudo-CB coordinate
            ca = residue["CA"].get_vector()
            ca_pcb = ca + pcb[0]
            return ca_pcb
        else:
            # If GLY's CA, N, or C is missing
            return None

    def _pcb_atom_from_non_gly(self, residue):
        """TODO"""

        residue_id = residue.id[1]

        if residue.get_resname() == "GLY":
            raise ValueError(f"Residue cannot be GLY.")
        else:
            # Get residue before and after input residue
            try:
                residue_before = self._residue_from_residue_id(residue_id - 1)
                residue_after = self._residue_from_residue_id(residue_id + 1)
            # If residue before or after do not exist, return None
            except KeyError:
                return None

            # Get pseudo-CB for non-GLY residue
            pcb = self._hse_ca._get_cb(residue_before, residue, residue_after)
            return pcb

    def _pcb_atom_from_gly(self, residue):
        """TODO"""

        if residue.get_resname() != "GLY":
            raise ValueError(f"Residue must be GLY, but is {residue.get_resname()}.")
        else:
            # Get pseudo-CB for GLY (vector centered at origin)
            pcb = self._hse_cb._get_gly_cb_vector(residue)
            return pcb

    def _side_chain_representative(self, residue_id):
        """TODO"""

        # Get biopython residue object
        residue = self._residue_from_residue_id(residue_id)
        residue_name = residue.get_resname()

        try:
            atom_name = SIDE_CHAIN_REPRESENTATIVE[residue_name]
            atom = residue[atom_name]
            return atom
        except KeyError:
            return None

    def _residue_from_residue_id(self, residue_id):
        """TODO"""

        residues = list(self._data_complex.get_residues())
        residue = [residue for residue in residues if residue.get_id()[1] == residue_id]

        if len(residue) == 1:
            return residue[0]
        else:
            raise KeyError(f"{len(residue)} residues were found, but must be 1.")

    def center_of_mass(self, entity, geometric=False):
        """
        Calculates gravitic [default] or geometric center of mass of an Entity.
        Geometric assumes all masses are equal (geometric=True).

        Parameters
        ----------
        entity : Bio.PDB.Entity.Entity or list of Bio.PDB.Atom.Atom
            Contains atoms for which the center of mass / centroid needs to be calculated:
            a) Basic container object for PDB heirachy. Structure, Model, Chain and Residue are
            subclasses of Entity.
            b) List of container objects for atoms.

        geometric : bool
            Geometric assumes all masses are equal (geometric=True). Defaults to False.

        Returns
        -------
        list of floats
            Gravitic [default] or geometric center of mass of an Entity.

        References
        ----------
        Copyright (C) 2010, Joao Rodrigues (anaryin@gmail.com)
        This code is part of the Biopython distribution and governed by its license.
        Please see the LICENSE file that should have been included as part of this package.
        """

        # Structure, Model, Chain, Residue
        if isinstance(entity, Entity.Entity):
            atom_list = entity.get_atoms()
        # List of Atoms
        elif hasattr(entity, "__iter__") and [x for x in entity if x.level == "A"]:
            atom_list = entity
        # Some other weirdo object
        else:
            raise ValueError(
                f"Center of Mass can only be calculated from the following objects:\n"
                f"Structure, Model, Chain, Residue, list of Atoms."
            )

        masses = []
        positions = [[], [], []]  # [ [X1, X2, ..] , [Y1, Y2, ...] , [Z1, Z2, ...] ]

        for atom in atom_list:
            masses.append(atom.mass)

            for i, coord in enumerate(atom.coord.tolist()):
                positions[i].append(coord)

        # If there is a single atom with undefined mass complain loudly.
        if "ukn" in set(masses) and not geometric:
            raise ValueError(
                f"Some atoms don't have an element assigned.\n"
                f"Try adding them manually or calculate the geometrical center of mass instead."
            )

        if geometric:
            return [sum(coord_list) / len(masses) for coord_list in positions]
        else:
            w_pos = [[], [], []]
            for atom_index, atom_mass in enumerate(masses):
                w_pos[0].append(positions[0][atom_index] * atom_mass)
                w_pos[1].append(positions[1][atom_index] * atom_mass)
                w_pos[2].append(positions[2][atom_index] * atom_mass)

            return [sum(coord_list) / sum(masses) for coord_list in w_pos]
