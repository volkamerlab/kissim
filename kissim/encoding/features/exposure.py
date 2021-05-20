"""
kissim.encoding.features.exposure

Defines the exposure feature.
"""

import logging

import numpy as np
import pandas as pd

from kissim.encoding.features import BaseFeature
from kissim.definitions import EXPOSURE_RADIUS, EXPOSURE_RATIO_CUTOFFS

logger = logging.getLogger(__name__)


class SolventExposureFeature(BaseFeature):
    """
    Exposure to the solvent for each pocket residue.

    Attributes
    ----------
    name : str or int
        Name for structure encoding by this feature.
    _residue_ids : list of int
        Residue IDs.
    _residue_ixs : list of int
        Residue indices.
    _categories : list of float or None
        Pocket residues' solvent exposure categories.
    _ratio : list of float
        Exposure values: Ratio of CA atoms in upper sphere / full sphere.
    _ratio_ca : list of float
        CA exposures: Ratio of CA atoms in upper sphere / full sphere (based on HSExposureCA).
    _ratio_cb : list of float
        CA exposures: Ratio of CA atoms in upper sphere / full sphere  (based on HSExposureCB).

    Notes
    -----
    Exposure of a residue describes the ratio of CA atoms in the upper sphere half around the
    CA-CB vector divided by the all CA atoms (given a sphere radius).

    References
    ----------
    Hamelryck, "An Amino Acid Has Two Sides: A New 2D Measure Provides a Different View of Solvent
    Exposure", Proteins, 59:38–48 (2005).
    """

    def __init__(self):

        self.name = None
        self._residue_ids = None
        self._residue_ixs = None
        self._categories = None
        self._ratio = None
        self._ratio_ca = None
        self._ratio_cb = None

    @classmethod
    def from_pocket(cls, pocket, radius=EXPOSURE_RADIUS):  # pylint: disable=W0221
        """
        Get exposure for each pocket residue from a Biopython-based pocket object.

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython
            Biopython-based pocket object.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.

        Returns
        -------
        kissim.encoding.SolventExposureFeature
            Exposure feature object.
        """

        feature = cls()
        feature.name = pocket.name
        feature._residue_ids = pocket._residue_ids
        feature._residue_ixs = pocket._residue_ixs
        exposures = feature._get_exposures(pocket, radius)
        feature._ratio = exposures["exposure"].to_list()
        feature._ratio_ca = exposures["ca.exposure"].to_list()
        feature._ratio_cb = exposures["cb.exposure"].to_list()
        feature._categories = [feature._get_category(ratio) for ratio in feature._ratio]
        return feature

    @property
    def values(self):
        """
        Exposure features for pocket residues.

        Returns
        -------
        list of float
            Exposure for pocket residues.
        """

        return self._categories

    @property
    def details(self):
        """
        Exposure features for pocket residues (verbose).

        Returns
        -------
        pandas.DataFrame
            Side chain orientation features for pocket residues (rows) with the following columns:
            - "exposure": Exposure ratio
            - "ca.exposure": Exposure ratio (based on HSExposureCA)
            - "cb.exposure": Exposure ratio (based on HSExposureCB)
        """

        features = pd.DataFrame(
            {
                "residue.id": self._residue_ids,
                "exposure.category": self._categories,
                "exposure.ratio": self._ratio,
                "exposure.ratio_ca": self._ratio_ca,
                "exposure.ratio_cb": self._ratio_cb,
            },
            index=self._residue_ixs,
        )
        features.index.name = "residue.ix"
        return features

    def _get_exposures(self, pocket, radius=EXPOSURE_RADIUS):
        """
        Get half sphere exposure calculation based on CB and CA atoms for full molecule.

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython
            Biopython-based pocket object.
        radius : float
            Sphere radius to be used for half sphere exposure calculation.

        Returns
        -------
        pandas.DataFrame
            Half sphere exposure data for both HSExposureCA and HSExposureCB calculation
            (columns for both methods each: up, down, angle CB-CA-pCB, and exposure ratio)
            for each pocket residue (index: residue ID).
        """

        # Calculate exposure values
        exposures_cb = self._get_exposures_by_method(pocket, radius, method="HSExposureCB")
        exposures_ca = self._get_exposures_by_method(pocket, radius, method="HSExposureCA")

        # Join both exposures calculations
        exposures_both = exposures_ca.join(exposures_cb, how="outer")

        # Select exposure value (CB values except CA values if CB is missing)
        exposures_both["exposure"] = exposures_both.apply(
            lambda row: row["ca.exposure"] if np.isnan(row["cb.exposure"]) else row["cb.exposure"],
            axis=1,
        )

        return exposures_both

    @staticmethod
    def _get_exposures_by_method(pocket, radius=EXPOSURE_RADIUS, method="HSExposureCB"):
        """
        Get exposure values for a given Half Sphere Exposure method,
        i.e. HSExposureCA or HSExposureCB.
        The exposure is defined as the ratio between the number of CA atoms in the lower half
        sphere (down) and the total number of CA atoms in the full sphere
        (high exposure = CA atoms are predominantly in the lower half sphere).

        Parameters
        ----------
        pocket : kissim.io.PocketBioPython
            Biopython-based pocket object.
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

        methods = ["HSExposureCB", "HSExposureCA"]

        # Calculate exposure values
        if method == methods[0]:
            exposures = pocket.hse_cb
        elif method == methods[1]:
            exposures = pocket.hse_ca
        else:
            raise ValueError(f'Method {method} unknown. Please choose from: {", ".join(methods)}')

        # Define column names
        up = f"{method[-2:].lower()}.up"
        down = f"{method[-2:].lower()}.down"
        angle = f"{method[-2:].lower()}.angle_cb_ca_pcb"
        exposure = f"{method[-2:].lower()}.exposure"

        # Transform into DataFrame
        exposures = pd.DataFrame(exposures, index=[up, down, angle], dtype=float).transpose()
        exposures.insert(loc=0, column="residue.id", value=[i[1][1] for i in exposures.index])
        exposures.reset_index(drop=True, inplace=True)

        # Calculate exposure value: down / (up + down)
        exposures[exposure] = exposures[down] / (exposures[up] + exposures[down])

        # So far, exposures only contains data for non-None residues, however we need to add
        # these None residues to homogenize feature and residue list lengths.
        # Let's use a trick:
        # 1. Merge exposures and residues (PDB IDs/indices without NaN residue PDB IDs)
        # by residue PDB ID in order to add residue indices to residue PDB IDs
        exposures = exposures.merge(
            pocket.residues.dropna(axis=0, subset=["residue.id"]), how="left", on="residue.id"
        )
        # 2. Merge residues (PDB IDs/indices WITH NaN residue PDB IDs) and exposures
        # by residue indices in order to add also residues with NaN residue PDB IDs
        exposures = pocket.residues.merge(
            exposures.drop(["residue.id"], axis=1), how="left", on="residue.ix"
        )
        # 3. Set residue indices as index
        exposures = exposures.drop(["residue.id"], axis=1).set_index("residue.ix")

        return exposures

    def _get_category(self, ratio):
        """
        Transform a given solvent exposure ratio into a category value, which defines the side chain
        orientation towards the pocket:
        - low solvent exposure (category 1.0)
        - intermediate solvent exposure (category 2.0)
        - high solvent exposure (category 3.0)

        Parameters
        ----------
        vertex_angle : float or None
            Vertex angle between a residue's CA atom (vertex), side chain representative and pocket
            centroid. Ranges between 0.0 and 180.0.

        Returns
        -------
        float or None
            Category for side chain orientation towards pocket.
            None if any of the input vectors are None.
        """

        if np.isnan(ratio):
            return np.nan
        elif 0.0 <= ratio <= EXPOSURE_RATIO_CUTOFFS[0]:  # Low solvent exposure
            return 1.0
        elif (
            EXPOSURE_RATIO_CUTOFFS[0] < ratio <= EXPOSURE_RATIO_CUTOFFS[1]
        ):  # Intermediate solvent exposure
            return 2.0
        elif EXPOSURE_RATIO_CUTOFFS[1] < ratio <= 1.0:  # High solvent exposure
            return 3.0
        else:
            raise ValueError(
                f"Molecule {self.name}: Unknown solvent exposure ratio {ratio}. "
                f"Only values between 0.0 and 180.0 allowed."
            )
