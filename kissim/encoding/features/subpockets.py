"""
kissim.encoding.features.subpockets 

Defines the subpockets feature.
"""

import logging

from kissim.encoding.features import BaseFeature

logger = logging.getLogger(__name__)


class SubpocketsFeature(BaseFeature):
    """
    Distances between all subpocket and all residues in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
    _distances : dict of (str: list of float)
        TODO
    _moments : dict of (str: list of float)
        TODO
    """

    def __init__(self):
        self._residue_ids = None
        self._distances = None
        self._moments = None

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
