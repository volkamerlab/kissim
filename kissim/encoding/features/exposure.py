"""
kissim.encoding.features.exposure TODO
"""

import logging

from Bio.PDB import HSExposureCA, HSExposureCB
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExposureFeature:
    """
    Exposure for each residue in the KLIFS-defined kinase binding site of 85 pre-aligned residues.
    Exposure of a residue describes the ratio of CA atoms in the upper sphere half around the
    CA-CB vector divided by the all CA atoms (given a sphere radius).

    Attributes
    ----------
    features : pandas.DataFrame
        1 feature (columns) for 85 residues (rows).

    References
    ----------
    Hamelryck, "An Amino Acid Has Two Sides: ANew 2D Measure Provides a Different View of Solvent
    Exposure", PROTEINS: Structure, Function, and Bioinformatics 59:38–48 (2005).
    """

    def __init__(self):

        self.features = None
        self.features_verbose = None

    def from_molecule(self, molecule, chain, radius=12.0):
        """
        Get exposure for each residue of a molecule.

        Parameters
        ----------
        molecule : biopandas.mol2.pandas_mol2.PandasMol2 or biopandas.pdb.pandas_pdb.PandasPdb
            Content of mol2 or pdb file as BioPandas object.
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        """

        # Get exposure data for all molecule's residues calculated based on
        # HSExposureCA and HSExposureCB
        exposures_molecule = self.get_molecule_exposures(chain, radius)

        # Get residues IDs belonging to KLIFS binding site
        klifs_res_ids = molecule.df.groupby(by=["res_id", "klifs_id"], sort=False).groups.keys()
        klifs_res_ids = pd.DataFrame(klifs_res_ids, columns=["res_id", "klifs_id"])
        klifs_res_ids.set_index("res_id", inplace=True, drop=False)

        # Keep only KLIFS residues
        # i.e. remove non-KLIFS residues and add KLIFS residues that were skipped in exposure
        # calculation
        exposures = klifs_res_ids.join(exposures_molecule, how="left")

        # Set index (from residue IDs) to KLIFS IDs
        exposures.set_index("klifs_id", inplace=True, drop=True)

        # Add column with CB exposure values, but with CA exposure values if CB exposure values
        # are missing
        exposures["exposure"] = exposures.apply(
            lambda row: row.ca_exposure if np.isnan(row.cb_exposure) else row.cb_exposure, axis=1
        )

        self.features = pd.DataFrame(
            exposures.exposure, index=exposures.exposure.index, columns=["exposure"]
        )
        self.features_verbose = exposures

    def get_molecule_exposures(self, chain, radius=12.0):
        """
        Get half sphere exposure calculation based on CB and CA atoms for full molecule.

        Parameters
        ----------
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.

        Returns
        -------
        pandas.DataFrame
            Half sphere exposure data for both HSExposureCA and HSExposureCB calculation
            (columns for both methods each: up, down, angle CB-CA-pCB, and exposure ratio)
            for each molecule residue (index: residue ID).
        """

        # Calculate exposure values
        exposures_cb = self.get_molecule_exposure_by_method(chain, radius, method="HSExposureCB")
        exposures_ca = self.get_molecule_exposure_by_method(chain, radius, method="HSExposureCA")

        # Join both exposures calculations
        exposures_both = exposures_ca.join(exposures_cb, how="outer")

        return exposures_both

    @staticmethod
    def get_molecule_exposure_by_method(chain, radius=12.0, method="HSExposureCB"):
        """
        Get exposure values for a given Half Sphere Exposure method,
        i.e. HSExposureCA or HSExposureCB.

        Parameters
        ----------
        chain : Bio.PDB.Chain.Chain
            Chain from PDB file.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.
        method : str
            Half sphere exposure method name: HSExposureCA or HSExposureCB.

        Returns
        -------
        pandas.DataFrame
            Half sphere exposure data (columns: up, down, angle CB-CA-pCB, and exposure ratio)
            for each molecule residue (index: residue ID).
        """

        methods = "HSExposureCB HSExposureCA".split()

        # Calculate exposure values
        if method == methods[0]:
            exposures = HSExposureCB(chain, radius)
        elif method == methods[1]:
            exposures = HSExposureCA(chain, radius)
        else:
            raise ValueError(f'Method {method} unknown. Please choose from: {", ".join(methods)}')

        # Define column names
        up = f"{method[-2:].lower()}_up"
        down = f"{method[-2:].lower()}_down"
        angle = f"{method[-2:].lower()}_angle_CB-CA-pCB"
        exposure = f"{method[-2:].lower()}_exposure"

        # Transform into DataFrame
        exposures = pd.DataFrame(
            exposures.property_dict, index=[up, down, angle], dtype=float
        ).transpose()
        exposures.index = [i[1][1] for i in exposures.index]

        # Calculate exposure value: up / (up + down)
        exposures[exposure] = exposures[up] / (exposures[up] + exposures[down])

        return exposures
