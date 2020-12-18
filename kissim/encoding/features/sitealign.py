"""
kissim.encoding.feature.sitealign

Defines the SiteAlign features.
"""

import logging

import pandas as pd

from kissim.schema import NON_STANDARD_AMINO_ACID_CONVERSION
from kissim.definitions import SITEALIGN_FEATURES
from kissim.encoding.features import BaseFeature

logger = logging.getLogger(__name__)


class SiteAlignFeature(BaseFeature):
    """
    SiteAlign features for each pocket residue.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
    _residue_ixs : list of int
        Residue indices.
    _residue_names : list of str
        Residue names (3-letter code).
    _categories : list of float
        Feature categories for given SiteAlign feature.

    Notes
    -----
    SiteAlign features include size, hydrogen bond donors, hydrogen bond acceptors,
    charge, alipathic, and aromatic features.
    Each residue is assigned to a category per SiteAlign feature.

    References
    ----------
    Schalon et al., "A simple and fuzzy method to align and compare druggable ligand-binding
    sites", Proteins, 71:1755-78 (2008).
    """

    def __init__(self):

        self._residue_ids = None
        self._residue_ixs = None
        self._residue_names = None
        self._categories = None

    @classmethod
    def from_pocket(cls, pocket, feature_name):  # pylint: disable=W0221
        """
        Get SiteAlign features of a given type for each pocket residue.

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython
            Biopython-based pocket object.
        feature_name : str
            Feature name:
            - "hba": Hydrogen bond acceptor feature
            - "hbd": Hydrogen bond donor feature
            - "size": Size feature
            - "charge": Charge feature
            - "aliphatic": Aliphatic feature
            - "aromatic": Aromatic feature

        Returns
        -------
        kissim.encoding.SiteAlignFeature
            SiteAlign features object.
        """
        feature = cls()
        feature._residue_ids = pocket.residues["residue.id"].to_list()
        feature._residue_ixs = pocket.residues["residue.ix"].to_list()
        feature._residue_names = [
            pocket._data_complex[residue_id].resname for residue_id in feature._residue_ids
        ]
        feature._categories = [
            feature._residue_to_value(residue_name, feature_name)
            for residue_name in feature._residue_names
        ]
        return feature

    @property
    def values(self):
        """
        SiteAlign features for pocket residues.

        Returns
        -------
        list of float
            SiteAlign features for pocket residues.
        """

        return self._categories

    @property
    def details(self):
        """
        Get feature values for pocket residues by feature name.

        Parameters
        ----------
        pocket : kissim.io.PocketDataFrame
            DataFrame-based pocket object.
        feature_name : str
            Feature name.

        Returns
        -------
        pandas.DataFrame
            Residues (rows) with the following columns:
            - "residue.name": Residue name
            - "sitealign.category": Feature value
        """

        return pd.DataFrame(
            {"residue.name": self._residue_names, "sitealign.category": self._categories},
            index=self._residue_ids,
        )

    def _residue_to_value(self, residue_name, feature_name):
        """
        Get a feature value (by feature name) for a residue (by residue name).

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        feature_name : str
            Feature name.

        Returns
        -------
        float
            Feature value.
        """

        self._raise_invalid_feature_name(feature_name)
        try:
            feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]
        except KeyError:
            residue_name = self._convert_modified_residue(residue_name)
            if residue_name:
                feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]
            else:
                feature_value = None
        return feature_value

    def _raise_invalid_feature_name(self, feature_name):
        """
        Check if feature name is part of the SiteAlign feature definitions.

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        """

        if feature_name not in SITEALIGN_FEATURES.columns:
            raise KeyError(
                f"Feature {feature_name} does not exist. "
                f'Please choose from: {", ".join(SITEALIGN_FEATURES.columns)}'
            )

    def _convert_modified_residue(self, residue_name):
        """
        Convert a non-standard residue in a standard residue if possible (if not return None).

        Parameters
        ----------
        residue_name : str
            Three-letter code for non-standard residue.

        Returns
        -------
        str or None
            Three-letter code for converted standard residue or None if conversion not possible.
        """

        try:
            residue_name_new = NON_STANDARD_AMINO_ACID_CONVERSION[residue_name]
            logger.info(f"Non-standard residue {residue_name} is processed as {residue_name_new}.")
            return residue_name_new

        except KeyError:
            logger.info(f"Non-standard residue {residue_name} is set to None.")
            return None
