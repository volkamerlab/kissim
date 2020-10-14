"""
kissim.io.core

Defines a base pocket class.
"""

from opencadd.io import Biopython

from ..utils import enter_temp_directory


class Pocket:
    """
    Class defining the base for pockets objects. TODO
    """

    @classmethod
    def from_local(cls, local, structure_id):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    @classmethod
    def from_remote(cls, remote, structure_id):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    @classmethod
    def _from_backend(cls, backend, structure_id):
        """TODO"""
        raise NotImplementedError("Implement in your subclass!")

    def _get_dataframe(self, backend, structure_id):
        """TODO"""

        dataframe = backend.coordinates.to_dataframe(structure_id, "complex", "pdb")
        return dataframe

    def _get_biopython(self, backend, structure_id):
        """TODO"""

        with enter_temp_directory():
            filepath = backend.coordinates.to_pdb(structure_id, "complex")
            biopython = Biopython.from_file(filepath)
            return biopython

    def _get_pocket_residue_ids(self, backend, structure_id):
        """TODO"""

        pocket_residues = backend.pockets.by_structure_id(structure_id)
        pocket_residue_ids = pocket_residues["residue.id"].to_list()
        pocket_residue_ids = [i for i in pocket_residue_ids if i != "_"]
        return pocket_residue_ids