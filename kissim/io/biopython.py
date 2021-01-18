"""
kissim.io.biopython

Defines a Biopython-based pocket class.
"""

import logging

import pandas as pd
from Bio.PDB import HSExposure, Vector, Entity
from opencadd.io import Biopython
from opencadd.databases.klifs import setup_remote
from opencadd.structure.pocket import PocketBase

from ..definitions import SIDE_CHAIN_REPRESENTATIVE
from ..schema import STANDARD_AMINO_ACIDS, NON_STANDARD_AMINO_ACID_CONVERSION
from ..utils import enter_temp_directory

logger = logging.getLogger(__name__)


class PocketBioPython(PocketBase):
    """
    Class defining the Biopython-based pocket object.

    Attributes
    ----------
    name : str
        Name of protein.
    _residue_ids : list of int
        Pocket residue IDs.
    _residue_ixs : list of int
        Pocket residue indices.
    _data_complex : Bio.PDB.Chain.Chain
        Structural data for the full complex (not the pocket only).
    _hse_ca_complex : Bio.PDB.HSExposure.HSExposureCA
        CA exposures for the full complex (not the pocket only).
    _hse_cb_complex : Bio.PDB.HSExposure.HSExposureCB
        CB exposures for the full complex (not the pocket only).

    Properties
    ----------
    center
    ca_atoms
    pcb_atoms
    side_chain_representatives
    hse_ca
    hse_cb
    """

    def __init__(self):

        self.name = None
        self._residue_ids = None
        self._residue_ixs = None
        self._data_complex = None
        self._hse_ca_complex = None
        self._hse_cb_complex = None

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Get Biopython-based pocket object from a KLIFS structure ID.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        klifs_session : None or opencadd.databases.klifs.session.Session
            Local or remote KLIFS session. If None, generate new remote session.

        Returns
        -------
        kissim.io.PocketBioPython
            Biopython-based pocket object.
        """

        if klifs_session is None:
            klifs_session = setup_remote()
        pocket = cls()
        pocket.name = structure_klifs_id
        # Load structure as biopython Chain object
        structure = pocket._get_biopython(structure_klifs_id, klifs_session)
        # KLIFS PDB files contain only one model and one chain - get their IDs
        model_id = next(structure.get_models()).id
        chain_id = next(structure.get_chains()).id
        chain = structure[model_id][chain_id]
        pocket._data_complex = chain
        # Get pocket residues
        pocket._residue_ids, pocket._residue_ixs = pocket._get_pocket_residue_ids(
            structure_klifs_id, klifs_session
        )
        # Get HSE data
        try:
            pocket._hse_ca_complex = HSExposure.HSExposureCA(pocket._data_complex)
            pocket._hse_cb_complex = HSExposure.HSExposureCB(pocket._data_complex)
        except AttributeError as e:
            logger.error(
                f"AttributeError: {e} in Bio.PDB.Exposure for structure KLIFS ID {structure_klifs_id}"
            )
            if e.args[0] == "'NoneType' object has no attribute 'norm'":
                # If HSE cannot be calculated with this error message, it is most likely related
                # to https://github.com/volkamerlab/kissim/issues/27
                # Return None for this pocket, with will result in a None fingerprint
                return None
            else:
                # Other errors shall be raised!!!
                raise AttributeError(f"{e}")

        return pocket

    def _get_biopython(self, structure_klifs_id, klifs_session):
        """
        Get structural data for a complex from a KLIFS structure ID as Biopython Structure object.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        Bio.PDB.Structure.Structure
            Structural data for a complex.
        """

        if klifs_session._client:
            with enter_temp_directory():
                filepath = klifs_session.coordinates.to_pdb(structure_klifs_id, "complex")
                biopython = Biopython.from_file(filepath)
                return biopython
        else:
            filepath = klifs_session.structures.by_structure_klifs_id(structure_klifs_id)[
                "structure.filepath"
            ][0]
            filepath = klifs_session._path_to_klifs_download / filepath / "complex.pdb"
            biopython = Biopython.from_file(filepath)
            return biopython

    def _get_pocket_residue_ids(self, structure_klifs_id, klifs_session):
        """
        Get pocket residues.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        residue_ids : list of int
            Pocket residue PDB IDs.
        residue_ixs : list of int
            Pocket residues indices.
        """

        if klifs_session._client:
            residues = klifs_session.pockets.by_structure_klifs_id(structure_klifs_id)
        else:
            residues = klifs_session.pockets.by_structure_klifs_id(
                structure_klifs_id, extension="pdb"
            )
        residue_ids = residues["residue.id"].to_list()
        residue_ixs = residues["residue.klifs_id"].to_list()
        residue_ids, residue_ixs = self._format_residue_ids_and_ixs(
            residue_ids, residue_ixs, "set pocket residues"
        )
        return residue_ids, residue_ixs

    @property
    def center(self):
        """
        Pocket centroid.

        Returns
        -------
        Bio.PDB.vectors.Vector
            Coordinates for the pocket centroid.
        """

        ca_atoms = self.ca_atoms
        ca_atom_vectors = ca_atoms["ca.atom"].to_list()
        ca_atom_vectors = [i for i in ca_atom_vectors if i is not None]
        centroid = self.center_of_mass(ca_atom_vectors, geometric=False)
        centroid = Vector(centroid)

        return centroid

    @property
    def ca_atoms(self):
        """
        Pocket CA atoms.

        Returns
        -------
        pandas.DataFrame
            Pocket CA atoms (rows) with the following columns:
            - "residue.id": Residue ID
            - "ca.atom": CA atom (Bio.PDB.Atom.Atom)
            - "ca.vector": CA atom vector (Bio.PDB.vectors.Vector)
        """

        ca_atoms = []
        for residue_id in self._residue_ids:
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

        return ca_atoms.astype({"residue.id": "Int32"})

    @property
    def pcb_atoms(self):
        """
        Pocket pseudo-CB atoms.

        Returns
        -------
        pandas.DataFrame
            Pocket pseudo-CB atoms (rows) with the following columns:
            - "residue.id": Residue ID
            - "pcb.vector": Pseudo-CB atom vector (Bio.PDB.vectors.Vector)
        """

        pcb_atoms = []
        for residue_id in self._residue_ids:
            pcb_atom = self._pcb_atom(residue_id)
            pcb_atoms.append([residue_id, pcb_atom])
        pcb_atoms = pd.DataFrame(pcb_atoms, columns=["residue.id", "pcb.vector"])

        return pcb_atoms.astype({"residue.id": "Int32"})

    @property
    def side_chain_representatives(self):
        """
        Pocket residues' side chain representatives.

        Returns
        -------
        pandas.DataFrame
            Pocket residues' side chain representatives (rows) with the following columns:
            - "residue.id": Residue ID
            - "sc.atom": Side chain representative (Bio.PDB.Atom.Atom or None)
            - "sc.vector": Side chain representative vector (Bio.PDB.vectors.Vector or None)
        """

        sc_atoms = []
        for residue_id in self._residue_ids:
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

        return sc_atoms.astype({"residue.id": "Int32"})

    @property
    def hse_ca(self):
        """
        CA exposures for pocket residues.

        Returns
        -------
        dict of tuple: tuple
            CA exposures (values) for pocket residues (keys).
            Example key-value pair: ('A', (' ', 461, ' ')): (0, 16, 0.4655905486374442)

        Notes
        -----
        Keys and values follow the Biopython notation:
        - Key: (chain id, (hetero-flag, sequence identifier, insertion code))
          - chain id: str (chain ID)
          - hetero-flag: str ("H_" for hetero atoms, "W" for water)
          - sequence identifier: int (residue ID)
          - insertion code: str (mark insertions)
        - Value: (hse_up, hse_down, angle)
          - hse_up: int (number of CA atoms in upper sphere)
          - hse_down: int (number of CA atoms in lower sphere)
          - angle: float (angle between CA-CB and CA-pCB)
        """

        return {
            residue: exposure
            for residue, exposure in self._hse_ca_complex.property_dict.items()
            if residue[1][1] in self._residue_ids
        }

    @property
    def hse_cb(self):
        """
        CB exposures for pocket residues.

        Returns
        -------
        dict of tuple: tuple
            CA exposures (values) for pocket residues (keys).
            Example key-value pair: ('A', (' ', 461, ' ')): (2, 14, 0.0)

        Notes
        -----
        See notes in hse_ca property.
        """

        return {
            residue: exposure
            for residue, exposure in self._hse_cb_complex.property_dict.items()
            if residue[1][1] in self._residue_ids
        }

    def _ca_atom(self, residue_id):
        """
        Get the CA atom for a residue.

        Parameters
        ----------
        residue_id : int
            Residue ID.

        Returns
        -------
        Bio.PDB.Atom.Atom or None
            Residue's CA atom.
        """

        residue = self._residue_from_residue_id(residue_id)
        try:
            return residue["CA"]
        except (KeyError, TypeError):
            return None

    def _pcb_atom(self, residue_id):
        """
        Get the pseudo-CB coordinates for a residue.

        Parameters
        ----------
        residue_id : int
            Residue ID.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom coordinates.
        """

        # Get biopython residue object
        residue = self._residue_from_residue_id(residue_id)

        if residue is None:
            return None
        else:
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
        """
        Get the pseudo-CB atom for a non-GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            Residue (not GLY).

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom coordinates.
        """

        residue_id = residue.id[1]

        if residue.get_resname() == "GLY":
            raise ValueError(f"Residue cannot be GLY.")
        else:
            # Get residue before and after input residue
            residue_before = self._residue_from_residue_id(residue_id - 1)
            residue_after = self._residue_from_residue_id(residue_id + 1)

            # If residue before or after do not exist, return None
            if residue_before is None or residue_after is None:
                return None
            else:
                # Get pseudo-CB for non-GLY residue
                pcb = self._hse_ca_complex._get_cb(residue_before, residue, residue_after)
                # Keep only pseudo-CB coordinates (drop angle between CA-CB and CA-pCB)
                pcb = pcb[0]
                return pcb

    def _pcb_atom_from_gly(self, residue):
        """
        Get the pseudo-CB atom for a GLY residue.

        Parameters
        ----------
        residue : Bio.PDB.Residue.Residue
            GLY residue.

        Returns
        -------
        Bio.PDB.vectors.Vector or None
            Pseudo-CB atom coordinates.
        """

        if residue.get_resname() != "GLY":
            raise ValueError(f"Residue must be GLY, but is {residue.get_resname()}.")
        else:
            # Get pseudo-CB for GLY (vector centered at origin)
            pcb = self._hse_cb_complex._get_gly_cb_vector(residue)
            return pcb

    def _side_chain_representative(self, residue_id):
        """
        Get the side chain representative for a residue.

        Parameters
        ----------
        residue_id : int
            Residue ID.

        Returns
        -------
        Bio.PDB.Atom.Atom or None
            Side chain representative.
        """

        # Get biopython residue object
        residue = self._residue_from_residue_id(residue_id)
        if residue is None:
            return None

        # Get residue name
        residue_name = residue.get_resname()

        # Convert non-standard amino acids if applicable
        if residue_name not in STANDARD_AMINO_ACIDS:
            try:
                residue_name = NON_STANDARD_AMINO_ACID_CONVERSION[residue_name]
            except KeyError:
                return None

        # Get side chain representative
        try:
            atom_name = SIDE_CHAIN_REPRESENTATIVE[residue_name]
            atom = residue[atom_name]
            return atom
        except KeyError:
            return None

    def _residue_from_residue_id(self, residue_id):
        """
        Get the residue from a residue ID.

        Parameters
        ----------
        residue_id : int
            Residue ID.

        Returns
        -------
        Bio.PDB.Residue.Residue or None
            Residue (None if input residue ID is not known).
        """

        residues = list(self._data_complex.get_residues())
        residue = [residue for residue in residues if residue.get_id()[1] == residue_id]

        if len(residue) == 1:
            return residue[0]
        elif len(residue) == 0:
            return None
        else:
            raise KeyError(f"{len(residue)} residues were found, but must be 1 or 0.")

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
