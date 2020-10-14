"""
kissim.encoding.feature.sitealign
"""

import logging

import pandas as pd

from ...definitions import MODIFIED_RESIDUE_CONVERSION, SITEALIGN_FEATURES

logger = logging.getLogger(__name__)


class SiteAlignFeature:
    """TODO"""

    def __init__(self):

        self._residue_ids = None
        self._features = None

    @classmethod
    def from_pocket_dataframe(cls, pocket, feature_name):
        """TODO"""

        feature = cls()
        pocket_residues = feature._pocket_to_values(pocket, feature_name)
        feature._residue_ids = pocket_residues["residue.id"].to_list()
        feature._features = pocket_residues["feature"].to_list()
        return feature

    @property
    def features(self):
        """TODO"""

        features = pd.DataFrame(self._features, columns=["feature"], index=self._residue_ids)
        return features

    def _pocket_to_values(self, pocket, feature_name):
        """TODO"""

        pocket_residues = (
            pocket.data[["residue.id", "residue.name"]].drop_duplicates().reset_index(drop=True)
        )
        pocket_residues["feature"] = pocket_residues.apply(
            lambda x: self._residue_to_value(x["residue.name"], feature_name), axis=1
        )

        return pocket_residues

    def _residue_to_value(self, residue_name, feature_name):
        """
        Get feature value for residue's size and pharmacophoric features
        (i.e. number of hydrogen  bond donor, hydrogen bond acceptors, charge features,
        aromatic features or aliphatic features)
        (according to SiteAlign feature encoding).

        Parameters
        ----------
        residue_name : str
            Three-letter code for residue.
        feature_name : str
            Feature name.

        Returns
        -------
        int
            Residue's size value according to SiteAlign feature encoding.
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

        if feature_name not in SITEALIGN_FEATURES.columns:
            raise KeyError(
                f"Feature {feature_name} does not exist. "
                f'Please choose from: {", ".join(SITEALIGN_FEATURES.columns)}'
            )

    def _convert_modified_residue(self, residue_name):

        # TODO check that input is not standard residue

        try:
            residue_name_new = MODIFIED_RESIDUE_CONVERSION[residue_name]
            logger.info(f"Non-standard residue {residue_name} is processed as {residue_name_new}.")
            return residue_name_new

        except KeyError:
            logger.info(f"Non-standard residue {residue_name} is set to None.")
            return None
