"""
kissim.io.dataframe

Defines a DataFrame-based pocket class.
"""

import pandas as pd

from opencadd.databases.klifs import setup_local, setup_remote
from .core import Pocket


class PocketDataframe(Pocket):
    """
    Class defining the base for structural objects for this package. TODO
    """

    def __init__(self):

        self._data_complex = None
        self._pocket_residue_ids = None

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
        pocket._data_complex = pocket._get_dataframe(backend, structure_id)
        pocket._pocket_residue_ids = pocket._get_pocket_residue_ids(backend, structure_id)
        return pocket

    @property
    def data(self):
        """TODO"""

        return self._data_complex[self._data_complex["residue.id"].isin(self._pocket_residue_ids)]

    @property
    def residue_ids(self):
        """TODO"""

        return self._pocket_residue_ids

    @property
    def centroid(self):
        """TODO"""

        ca_atoms = self.ca_atoms
        centroid = ca_atoms[["atom.x", "atom.y", "atom.z"]].mean()
        return centroid.to_list()

    @property
    def ca_atoms(self):
        """TODO"""

        ca_atoms = []
        for residue_id, _ in self.data.groupby("residue.id", sort=False):
            ca_atom = self._ca_atom(residue_id)
            ca_atoms.append(ca_atom)
        ca_atoms = pd.concat(ca_atoms).reset_index(drop=True)
        return ca_atoms

    def _ca_atom(self, residue_id):
        """TODO"""

        data_complex = self._data_complex
        ca_atom = data_complex[
            (data_complex["atom.name"] == "CA") & (data_complex["residue.id"] == residue_id)
        ]
        if len(ca_atom) == 1:
            return ca_atom
        else:
            return None
