"""
kissim.encoding.features.core 

Defines the core classes and functions.
"""


class BaseFeature:
    """
    Base class for kissim features that encode the KLIFS-defined kinase binding site residues.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.

    Properties
    ----------
    values : list of float
        Feature values for pocket residues.
    details : pandas.DataFrame
        Feature details for pocket residues.
    """

    def __init__(self):
        self._residue_ids = None

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, remote=None):
        """
        Get feature from a KLIFS structure ID.
        TODO At the moment only remotely, in the future allow also locally.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        remote : None or opencadd.databases.klifs.session.Session
            Remote KLIFS session. If None, generate new remote session.
        """
        raise NotImplementedError("Implement in your subclass!")

    @classmethod
    def from_pocket(cls, pocket):
        """
        Get feature from a pocket object.

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython or kissim.io.PocketDataFrame
            Pocket object.
        """
        raise NotImplementedError("Implement in your subclass!")

    @property
    def values(self):
        """
        Feature values for pocket residues.

        Returns
        -------
        list of float
            Features for pocket residues.
        """
        raise NotImplementedError("Implement in your subclass!")

    @property
    def details(self):
        """
        Feature details for pocket residues.

        Returns
        -------
        pandas.DataFrame
            Feature details for pocket residues.
        """
        raise NotImplementedError("Implement in your subclass!")