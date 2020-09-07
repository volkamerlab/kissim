"""
TODO
"""

import logging

import pandas as pd

from ..definitions import MODIFIED_RESIDUE_CONVERSION, SITEALIGN_FEATURES

logger = logging.getLogger(__name__)


class PharmacophoreSizeFeatures:
    """
    Pharmacophore and size features for each residue in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues, as described by SiteAlign (Schalon et al. Proteins. 2008).

    Pharmacophoric features include hydrogen bond donor, hydrogen bond acceptor, aromatic,
    aliphatic and charge feature.

    Attributes
    ----------
    features : pandas.DataFrame
        6 features (columns) for 85 residues (rows).

    References
    ----------
    Schalon et al., "A simple and fuzzy method to align and compare druggable ligand‐binding
    sites", Proteins, 2008.
    """

    def __init__(self):

        self.molecule_code = None
        self.features = None

    def from_molecule(self, molecule):
        """
        Get pharmacophoric and size features for each residues of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.

        Returns
        -------
        pandas.DataFrame
            Pharmacophoric and size features (columns) for each residue = KLIFS position (rows).
        """

        self.molecule_code = molecule.code

        feature_matrix = []

        for feature_name in SITEALIGN_FEATURES.columns:

            # Select from DataFrame first row per KLIFS position (index) and residue name
            residues = molecule.df.groupby(by="klifs_id", sort=False).first()["res_name"]

            # Get feature values for each KLIFS position
            features = residues.apply(lambda residue: self.from_residue(residue, feature_name))
            features.rename(feature_name, inplace=True)

            feature_matrix.append(features)

        features = pd.concat(feature_matrix, axis=1)

        self.features = features

    def from_residue(self, residue_name, feature_name):
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

        if feature_name not in SITEALIGN_FEATURES.columns:
            raise KeyError(
                f"Feature {feature_name} does not exist. "
                f'Please choose from: {", ".join(SITEALIGN_FEATURES.columns)}'
            )

        try:

            feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]

        except KeyError:

            if residue_name in MODIFIED_RESIDUE_CONVERSION.keys():

                logger.info(
                    f"{self.molecule_code}, {feature_name} feature: "
                    f"Non-standard amino acid {residue_name} is processed as "
                    f"{MODIFIED_RESIDUE_CONVERSION[residue_name]}."
                )

                residue_name = MODIFIED_RESIDUE_CONVERSION[residue_name]
                feature_value = SITEALIGN_FEATURES.loc[residue_name, feature_name]

            else:

                logger.info(
                    f"{self.molecule_code}, {feature_name} feature: "
                    f"Non-standard amino acid {residue_name} is set to None."
                )

                feature_value = None

        return feature_value
