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
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session):
        """
        Get pocket object from a KLIFS structure ID (locally).

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        kissim.io.PocketDataframe or kissim.io.PocketBiopython
            Pocket object.
        """
        raise NotImplementedError("Implement in your subclass!")

    def _get_dataframe(self, structure_klifs_id, klifs_session):
        """
        Get structural data for a complex from a KLIFS structure ID as DataFrame.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : opencadd.databases.klifs.session.Session
            Local or remote KLIFS session.

        Returns
        -------
        pandas.DataFrame
            Structural data for a complex.
        """

        dataframe = klifs_session.coordinates.to_dataframe(structure_klifs_id, "complex", "pdb")
        return dataframe

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
        pandas.DataFrame
            Structural data for a complex.
        """

        with enter_temp_directory():
            filepath = klifs_session.coordinates.to_pdb(structure_klifs_id, "complex")
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
        list of str
            Pocket residues.
        """

        pocket_residues = klifs_session.pockets.by_structure_klifs_id(structure_klifs_id)
        pocket_residue_ids = pocket_residues["residue.id"].to_list()
        pocket_residue_ids = [i for i in pocket_residue_ids if i != "_"]
        return pocket_residue_ids