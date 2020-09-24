"""
kissim.io.biopython.pocket

Defines a basic class for structural objects for this package.
"""

from opencadd.databases.klifs import setup_remote

from .core import StructureBiopython


class PocketBiopython(StructureBiopython):
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
        structure._set_residue_pdb_ids(structure_id)
        structure._set_hse()

        filepath.unlink()

        return structure

    def _set_residue_pdb_ids(self, structure_id):
        """TODO"""

        remote = setup_remote()
        pocket_residues = remote.pockets.from_structure_id(structure_id)
        pocket_residue_pdb_ids = pocket_residues["residue.pdb_id"].to_list()
        pocket_residue_pdb_ids = [int(i) for i in pocket_residue_pdb_ids if i != "_"]
        self._residue_pdb_ids = pocket_residue_pdb_ids