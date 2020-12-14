"""
kissim.encoding.features.exposure

Defines the exposure feature.
"""

import logging

import numpy as np
import pandas as pd

from kissim.io import PocketBioPython
from kissim.encoding.features import BaseFeature

logger = logging.getLogger(__name__)


class SolventExposureFeature(BaseFeature):
    """
    Exposure to the solvent for each residue in the KLIFS-defined kinase binding site of 85
    pre-aligned residues.

    Attributes
    ----------
    _residue_ids : list of int
        Residue IDs.
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

        self._residue_ids = None
        self._ratio = None
        self._ratio_ca = None
        self._ratio_cb = None

    @classmethod
    def from_structure_klifs_id(cls, structure_id, remote=None):
        """
        Get exposure for each pocket residue from a KLIFS structure ID.
        TODO At the moment only remotely, in the future allow also locally.

        Parameters
        ----------
        structure_id : int
            KLIFS structure ID.
        remote : None or opencadd.databases.klifs.session.Session
            Remote KLIFS session. If None, generate new remote session.

        Returns
        -------
        kissim.encoding.SolventExposureFeature
            Exposure feature object.
        """

        pocket_biopython = PocketBioPython.from_remote(structure_id, remote)
        feature = cls.from_pocket(pocket_biopython)
        return feature

    @classmethod
    def from_pocket(cls, pocket, radius=12.0):
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
        feature._residue_ids = pocket.residue_ids
        exposures = feature._get_exposures(pocket, radius)
        feature._ratio = exposures["exposure"].to_list()
        feature._ratio_ca = exposures["ca.exposure"].to_list()
        feature._ratio_cb = exposures["cb.exposure"].to_list()
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

        return self._ratio

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
                "exposure.ratio": self._ratio,
                "exposure.ratio_ca": self._ratio_ca,
                "exposure.ratio_cb": self._ratio_cb,
            },
            index=self._residue_ids,
        )
        return features

    def _get_exposures(self, pocket, radius=12.0):
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
    def _get_exposures_by_method(pocket, radius=12.0, method="HSExposureCB"):
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

        # Select pocket residues only

        # Define column names
        up = f"{method[-2:].lower()}.up"
        down = f"{method[-2:].lower()}.down"
        angle = f"{method[-2:].lower()}.angle_cb_ca_pcb"
        exposure = f"{method[-2:].lower()}.exposure"

        # Transform into DataFrame
        exposures = pd.DataFrame(exposures, index=[up, down, angle], dtype=float).transpose()
        exposures.index = [i[1][1] for i in exposures.index]

        # Calculate exposure value: down / (up + down)
        exposures[exposure] = exposures[down] / (exposures[up] + exposures[down])

        return exposures
