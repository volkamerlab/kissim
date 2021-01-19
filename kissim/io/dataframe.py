"""
kissim.io.dataframe

Defines a DataFrame-based pocket class.
"""

from opencadd.structure.pocket import Pocket

from . import KlifsToKissimData


class PocketDataFrame(Pocket):
    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Get DataFrame-based pocket object from a KLIFS structure ID.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        klifs_session : None or opencadd.databases.klifs.session.Session
            Local or remote KLIFS session. If None, generate new remote session.

        Returns
        -------
        kissim.io.PocketDataFrame or None
            DataFrame-based pocket object.
        """

        data = KlifsToKissimData.from_structure_klifs_id(structure_klifs_id, klifs_session)
        if data:
            pocket = cls.from_text(
                data.text, data.extension, data.residue_ids, data.residue_ixs, structure_klifs_id
            )
            return pocket
        else:
            return None
