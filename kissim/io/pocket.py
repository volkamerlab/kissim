"""
kissim.io.pocket

Defines classes that convert structural data into kissim pocket objects.
"""

from opencadd.databases.klifs import setup_remote
from .core import Base
from .complex import Complex


class Pocket(Base):
    """
    Class defining a pocket structure object.
    """

    @classmethod
    def from_structure_id(cls, structure_id):
        """
        Load pocket from KLIFS structure ID.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.

        Returns
        -------
        kissim.core.Pocket
            Kissim pocket object.
        """

        # Get complex object
        complex = Complex.from_structure_id(structure_id)

        # Get pocket residues
        remote = setup_remote()
        pocket_residues = remote.pockets.from_structure_id(structure_id)
        pocket_residue_pdb_ids = pocket_residues["residue.pdb_id"].to_list()
        pocket_residue_pdb_ids = [int(i) for i in pocket_residue_pdb_ids if i != "_"]

        # Select residues in DataFrame
        dataframe = complex.dataframe
        dataframe = dataframe[dataframe["residue.pdb_id"].isin(pocket_residue_pdb_ids)]

        # Select residues in biopython Structure object
        _biopython = None

        # Select residues in biopython HSEExposure object
        _hse = complex._hse
        _hse = {residue: hse for residue, hse in _hse.items() if residue in pocket_residue_pdb_ids}

        pocket = cls(dataframe, _biopython, _hse)

        return pocket
