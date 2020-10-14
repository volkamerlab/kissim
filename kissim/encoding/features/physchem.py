"""
TODO
"""
import logging

import pandas as pd

from . import SideChainOrientationFeature, ExposureFeature

logger = logging.getLogger(__name__)


class PhysicoChemicalFeatures:
    """
    Physicochemical features for each residue in the KLIFS-defined kinase binding site
    of 85 pre-aligned residues.

    Physicochemical properties:
    - Size
    - Pharmacophoric features:
      Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Half sphere exposure

    Attributes
    ----------
    features : pandas.DataFrame
        8 features (columns) for 85 residues (rows).
    """

    def __init__(self):

        self.features = None
        self.features_verbose = {"side_chain_orientation": None, "exposure": None}

    def from_molecule(self, molecule, chain):
        """
        Get physicochemical properties for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        """

        pharmacophore_size = PharmacophoreSizeFeatures()
        pharmacophore_size.from_molecule(molecule)

        side_chain_orientation = SideChainOrientationFeature()
        side_chain_orientation.from_molecule(molecule, chain)

        exposure = ExposureFeature()
        exposure.from_molecule(molecule, chain, radius=12.0)

        # Concatenate all physicochemical features
        physicochemical_features = pd.concat(
            [pharmacophore_size.features, side_chain_orientation.features, exposure.features],
            axis=1,
        )

        # Bring all fingerprints to same dimensions
        # (i.e. add currently missing residues in DataFrame)
        empty_df = pd.DataFrame([], index=range(1, 86))
        physicochemical_features = pd.concat([empty_df, physicochemical_features], axis=1)

        # Set all None to nan
        physicochemical_features.fillna(value=pd.np.nan, inplace=True)

        self.features = physicochemical_features

        self.features_verbose["side_chain_orientation"] = side_chain_orientation.features_verbose
        self.features_verbose["exposure"] = exposure.features_verbose
