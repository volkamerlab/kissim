"""
kissim.io.biopython

Defines a basic class for structural objects for this package.
"""

from opencadd.databases.klifs import setup_remote

from .core import StructureBiopython


class ComplexBiopython(StructureBiopython):
    """
    Class defining the base for structural objects for this package. TODO
    """

    def __init__(self):

        self._data = None
        self._residue_pdb_ids = None
        self._hse_ca = None
        self._hse_cb = None

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
        structure._set_hse()

        filepath.unlink()

        return structure

    def _set_residue_pdb_ids(self):
        """TODO"""

        residues = list(self._data.get_residues())
        residue_pdb_ids = [residue.get_id()[1] for residue in residues]
        self._residue_pdb_ids = residue_pdb_ids