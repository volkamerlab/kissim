"""
kissim.encoding.feature.sitealign

Defines the SiteAlign features: Size, hydrogen bond donor, hydrogen bond acceptors, charge, 
aliphatic, and aromatic.
"""

import logging

import pandas as pd

from ...definitions import MODIFIED_RESIDUE_CONVERSION, SITEALIGN_FEATURES

logger = logging.getLogger(__name__)


class SiteAlignFeature:
    """
    SiteAlign features for each residue: Size, hydrogen bond donors, hydrogen bond acceptors,
    charge, alipathic, and aromatic features.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
    _values : dict (str: list of float)
        Feature values (dict values) for different SiteAlign features (dict key).

    References
    ----------
    Schalon et al., "A simple and fuzzy method to align and compare druggable ligand-binding
    sites", Proteins, 71:1755-78 (2008).
    """

    def __init__(self):

        self._residue_ids = None
        self._values = {
            "hba": None,
            "hbd": None,
            "size": None,
            "charge": None,
            "aliphatic": None,
            "aromatic": None,
        }

    @classmethod
    def from_pocket_dataframe(cls, pocket):
        """
        Get SiteAlign features for each residue of a pocket.

        Parameters
        ----------
        pocket : kissim.io.biopython.pocket.PocketDataFrame
            DataFrame-based pocket object.

        Returns
        -------
        kissim.encoding.features.SiteAlignFeature
            SiteAlign features object.
        """

        feature = cls()
        feature._residue_ids = pocket.data["residue.id"].drop_duplicates().to_list()
        for feature_name, values in feature._values.items():
            values = feature._pocket_to_values(pocket, feature_name)
            values = values["feature"].to_list()
            feature._values[feature_name] = values
        return feature

    @property
    def features(self):
        """
        SiteAlign features for pocket residues.

        Returns
        -------
        pandas.DataFrame
            SiteAlign features for pocket residues (index).
        """

        features = pd.DataFrame(self._values, index=self._residue_ids)
        return features

    def _pocket_to_values(self, pocket, feature_name):
        """
        Get feature values for pocket residues by feature name.

        Parameters
        ----------
        pocket : kissim.io.biopython.pocket.PocketDataFrame
            DataFrame-based pocket object.
        feature_name : str
            Feature name.

        Returns
        -------
        pandas.DataFrame
            Residues (rows) with the following columns:
            - "residue.id": Residue ID
            - "residue.name": Residue name
            - "feature": Feature value
        """

        pocket_residues = (
            pocket.data[["residue.id", "residue.name"]].drop_duplicates().reset_index(drop=True)
        )
        pocket_residues["feature"] = pocket_residues.apply(
            lambda x: self._residue_to_value(x["residue.name"], feature_name), axis=1
        )
        return pocket_residues

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

        self.check_valid_feature_name(feature_name)
        try:
            feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]
        except KeyError:
            residue_name = _convert_modified_residue(residue_name)
            if residue_name:
                feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]
            else:
                feature_value = None
        return feature_value

    def check_valid_feature_name(self, feature_name):
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
            residue_name_new = MODIFIED_RESIDUE_CONVERSION[residue_name]
            logger.info(f"Non-standard residue {residue_name} is processed as {residue_name_new}.")
            return residue_name_new

        except KeyError:
            logger.info(f"Non-standard residue {residue_name} is set to None.")
            return None
