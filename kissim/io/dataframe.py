"""
kissim.io.dataframe

Defines a DataFrame-based pocket class.
"""

import pandas as pd
from opencadd.databases.klifs import setup_remote

from .core import Pocket


class PocketDataFrame(Pocket):
    """
    Class defining the DataFrame-based pocket object.

    Attributes
    ----------
    _residue_ids : list of str
        Pocket residue IDs.
    _data_complex : pandas.DataFrame
        Structural data for the full complex (not the pocket only).
    """

    def __init__(self):

        self._residue_ids = None
        self._data_complex = None

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Get DataFrame-based pocket object from a KLIFS structure ID (remotely).

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : None or opencadd.databases.klifs.session.Session
            Local or remote KLIFS session. If None, generate new remote session.

        Returns
        -------
        kissim.io.PocketDataframe
            DataFrame-based pocket object.
        """

        if not klifs_session:
            klifs_session = setup_remote()
        pocket = cls()
        pocket._data_complex = pocket._get_dataframe(structure_klifs_id, klifs_session)
        pocket._residue_ids = pocket._get_pocket_residue_ids(structure_klifs_id, klifs_session)
        return pocket

    @property
    def data(self):
        """
        Pocket data.

        Returns
        -------
        pandas.DataFrame
            Pocket atoms (rows) with the following columns:
            - "atom.id" and "atom.name": Atom ID and name
            - "atom.x", "atom.y", "atom.z": Atom x/y/z coordinates
            - "residue.id" and "residue.name": Residue ID and name
            - "residue.klifs_id": Residue KLIFS ID
            - "residue.klifs_region_id": Residue KLIFS region ID (e.g. xDFG.80)
            - "residue.klifs_region": Residue KLIFS region (e.g. xDFG)
            - "residue.klifs_color": Residue KLIFS color
        """

        data_pocket = self._data_complex[self._data_complex["residue.id"].isin(self._residue_ids)]

        # Check if number of residues correct?
        n_residues_data = data_pocket["residue.id"].unique().shape[0]
        n_residue_ids = len(self._residue_ids)
        if not n_residues_data == n_residue_ids:
            raise ValueError(
                f"PocketDataFrame: "
                f"Number of residues is unequal in the properties `data` ({n_residues_data}) "
                f"and `residue_ids` ({n_residue_ids})"
            )

        return data_pocket

    @property
    def residue_ids(self):
        """
        Pocket residue IDs.

        Returns
        -------
        list of str
            Pocket residue IDs.
        """

        return self._residue_ids

    @property
    def centroid(self):
        """
        Pocket centroid.

        Returns
        -------
        list of float
            Coordinates for the pocket centroid.
        """

        ca_atoms = self.ca_atoms
        centroid = ca_atoms[["atom.x", "atom.y", "atom.z"]].mean()
        return centroid.to_list()

    @property
    def ca_atoms(self):
        """
        Pocket CA atoms.

        Returns
        -------
        pandas.DataFrame
            Pocket CA atoms (rows) with the following columns:
            - "atom.id" and "atom.name": Atom ID and name
            - "atom.x", "atom.y", "atom.z": Atom x/y/z coordinates
            - "residue.id" and "residue.name": Residue ID and name
            - "residue.klifs_ids": Residue KLIFS ID
            - "residue.klifs_region_id": Residue KLIFS region ID (e.g. xDFG.80)
            - "residue.klifs_region": Residue KLIFS region (e.g. xDFG)
            - "residue.klifs_color": Residue KLIFS color
        """

        ca_atoms = []
        for residue_id, _ in self.data.groupby("residue.id", sort=False):
            ca_atom = self._ca_atom(residue_id)
            ca_atoms.append(ca_atom)
        ca_atoms = pd.concat(ca_atoms).reset_index(drop=True)
        return ca_atoms

    def _ca_atom(self, residue_id):
        """
        Pocket CA atom.

        Returns
        -------
        pandas.DataFrame
            Pocket CA atom (1 row) with the following columns:
            - "atom.id" and "atom.name": Atom ID and name
            - "atom.x", "atom.y", "atom.z": Atom x/y/z coordinates
            - "residue.id" and "residue.name": Residue ID and name
            - "residue.klifs_ids": Residue KLIFS ID
            - "residue.klifs_region_id": Residue KLIFS region ID (e.g. xDFG.80)
            - "residue.klifs_region": Residue KLIFS region (e.g. xDFG)
            - "residue.klifs_color": Residue KLIFS color
        """

        data_complex = self._data_complex
        ca_atom = data_complex[
            (data_complex["atom.name"] == "CA") & (data_complex["residue.id"] == residue_id)
        ]
        if len(ca_atom) == 1:
            return ca_atom
        else:
            return None
