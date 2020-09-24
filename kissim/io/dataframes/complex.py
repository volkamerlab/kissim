"""
kissim.io.dataframes.complex

Defines a basic class for structural objects for this package.
"""

from opencadd.databases.klifs import setup_remote

from .core import StructureDataframe


class ComplexDataframe(StructureDataframe):
    """
    Class defining the base for structural objects for this package. TODO
    """

    def from_file(self, filepath):
        """TODO"""

        # Get structure ID from filepath

        print("Dummy function!")
        structure_id = 12347  # TODO
        self.from_structure_id(structure_id)

    @classmethod
    def from_structure_id(cls, structure_id):
        """TODO"""

        remote = setup_remote()
        filepath = remote.coordinates.to_file(structure_id, ".", "complex", "pdb")

        structure = cls()
        structure._set_data(filepath)
        structure._set_residue_pdb_ids()

        filepath.unlink()

        return structure

    def _set_residue_pdb_ids(self):
        """TODO"""

        residue_pdb_ids = list(self._data["residue.pdb_id"].unique())
        self._residue_pdb_ids = residue_pdb_ids
