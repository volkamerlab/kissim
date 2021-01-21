"""
kissim.encoding.features.base 

Defines the core classes and functions.
"""


class BaseFeature:
    """
    Base class for features that encode pocket residues.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
    _residue_ixs : list of int
        Residue indices.

    Properties
    ----------
    name : str or int
        Name for structure encoding by this feature.
    values : list of float
        Feature values for pocket residues.
    details : pandas.DataFrame
        Feature details for pocket residues.
    """

    def __init__(self):
        self.name = None
        self._residue_ids = None
        self._residue_ixs = None

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Get feature from a KLIFS structure ID.

        Parameters
        ----------
        structure_klifs_id : int
            KLIFS structure ID.
        klifs_session : None or opencadd.databases.klifs.session.Session
            Local or remote KLIFS session. If None, generate new remote session.
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
