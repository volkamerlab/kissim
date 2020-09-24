"""
kissim.io.dataframes.core

Defines a basic class for structural objects for this package.
"""

import pandas as pd

from opencadd.io import DataFrame


class StructureDataframe:
    """
    Class defining the base for structural objects for this package. TODO
    """

    def __init__(self):

        self._data = None
        self._residue_pdb_ids = None

    def from_file(self, filepath):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    def from_structure_id(self, structure_id):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    def _set_data(self, filepath):
        """TODO"""

        data = DataFrame.from_file(filepath)
        self._data = data

    def _set_residue_pdb_ids(self, structure_id):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    @property
    def data(self):
        """TODO"""
        data = self._data
        data = data[data["residue.pdb_id"].isin(self.residue_pdb_ids)]
        return data

    @property
    def residue_pdb_ids(self):
        """TODO"""
        return self._residue_pdb_ids

    @property
    def centroid(self):
        """TODO"""

        ca_atoms = self.ca_atoms
        centroid = ca_atoms[["atom.x", "atom.y", "atom.z"]].mean()

        return centroid.to_list()

    @property
    def ca_atoms(self):
        """TODO"""
        data = self.data

        ca_atoms = []
        for residue_pdb_id, _ in data.groupby("residue.pdb_id", sort=False):
            ca_atom = self.ca_atom(residue_pdb_id)
            ca_atoms.append(ca_atom)
        ca_atoms = pd.concat(ca_atoms).reset_index(drop=True)

        return ca_atoms

    def ca_atom(self, residue_pdb_id):
        """TODO"""

        data = self.data

        ca_atom = data[(data["atom.name"] == "CA") & (data["residue.pdb_id"] == residue_pdb_id)]

        if len(ca_atom) == 1:
            return ca_atom
        else:
            return None
