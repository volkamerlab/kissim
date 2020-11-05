"""
kissim.io.dataframe

Defines a DataFrame-based pocket class.
"""

import pandas as pd
from opencadd.databases.klifs import setup_remote

from .core import Pocket


class PocketDataframe(Pocket):
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
    def from_remote(cls, structure_id, remote=None):
        """
        Get DataFrame-based pocket object from a KLIFS structure ID (remotely).

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        remote : None or opencadd.databases.klifs.session.Session
            Remote KLIFS session. If None, generate new remote session.

        Returns
        -------
        kissim.io.dataframe.PocketDataframe
            DataFrame-based pocket object.
        """

        if not remote:
            remote = setup_remote()
        return cls._from_backend(remote, structure_id)

    @classmethod
    def _from_backend(cls, backend, structure_id):
        """
        Get DataFrame-based pocket object from a KLIFS structure ID.

        Parameters
        ----------
        backend : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.
        structure_id : int
            KLIFS structure ID.

        Returns
        -------
        kissim.io.dataframe.PocketDataframe
            DataFrame-based pocket object.
        """

        pocket = cls()
        pocket._data_complex = pocket._get_dataframe(backend, structure_id)
        pocket._residue_ids = pocket._get_pocket_residue_ids(backend, structure_id)
        return pocket

    @property
    def data(self):
        """
        Pocket atoms.

        Returns
        -------
        pandas.DataFrame
            Pocket atoms (rows) with the following columns:
            - "atom.id" and "atom.name": Atom ID and name
            - "atom.x", "atom.y", "atom.z": Atom x/y/z coordinates
            - "residue.id" and "residue.name": Residue ID and name
            - "residue.klifs_ids": Residue KLIFS ID
            - "residue.klifs_region_id": Residue KLIFS region ID (e.g. xDFG.80)
            - "residue.klifs_region": Residue KLIFS region (e.g. xDFG)
            - "residue.klifs_color": Residue KLIFS color
        """

        return self._data_complex[self._data_complex["residue.id"].isin(self._residue_ids)]

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
