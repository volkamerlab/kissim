"""
kissim.encoding.fingerprint

Defines the kissim fingerprint.
"""

import logging

import numpy as np
import pandas as pd
from opencadd.databases.klifs import setup_remote

from kissim.io import PocketBioPython, PocketDataFrame
from kissim.encoding.features import (
    SiteAlignFeature,
    SideChainOrientationFeature,
    SolventExposureFeature,
    SubpocketsFeature,
)

logger = logging.getLogger(__name__)


class Fingerprint:
    """
    The KLIFS kinase pocket is defined by 85 pre-aligned residues, which are described each with
    (i) 8 physicochemical and
    (ii) 4 distance features as well as
    (iii) the first three moments of aforementioned feature distance distributions.
    Fingerprints can consist of all or a subset of these three feature types.


    Attributes
    ----------
    structure_klifs_id : str
        Structure KLIFS ID.
    values : dict of pandas.DataFrame
        Fingerprint values, consisting of physicochemical, distance and moment features.

    Notes
    -----
    PHYSICOCHEMICAL features (85 x 8 matrix = 680 bits):

    - Size
    - Pharmacophoric features:
      Hydrogen bond donor, hydrogen bond acceptor, aromatic, aliphatic and charge feature
    - Side chain orientation
    - Solvent exposure

    SPATIAL features:

    - DISTANCE of each residue to 4 reference points (85 x 4 matrix = 340 bits):
      - Pocket center
      - Hinge region (subpocket center)
      - DFG region (subpocket center)
      - Front pocket (subpocket center)
    - MOMENTS for distance distributions for the 4 reference points (4 x 3 matrix = 12 bits):
      - Moment 1: Mean
      - Moment 2: Standard deviation
      - Moment 3: Skewness (cube root)

    The terminology used for the feature hierarchy is the following:
    - Feature category, e.g. spatial or physicochemical
    - Feature type, e.g. distance to centroid or size
    """

    def __init__(self):

        self.structure_klifs_id = None
        self.values_dict = None
        self.residue_ids = None
        self.residue_ixs = None

    @property
    def physicochemical(self):
        """TODO"""
        features = self.values_dict["physicochemical"]
        features = pd.DataFrame(features, index=self.residue_ixs)
        features.index.name = "residue.ix"
        return features

    @property
    def distances(self):
        """TODO"""
        features = self.values_dict["spatial"]["distances"]
        features = pd.DataFrame(features, index=self.residue_ixs)
        features.index.name = "residue.ix"
        return features

    @property
    def moments(self):
        """TODO"""
        features = self.values_dict["spatial"]["moments"]
        features = pd.DataFrame(features, index=[1, 2, 3])
        features.index.name = "moments"
        return features

    @classmethod
    def from_structure_klifs_id(cls, structure_klifs_id, klifs_session=None):
        """
        Calculate fingerprint for a KLIFS structure (by structure KLIFS ID).

        Parameters
        ----------
        structure_klifs_id : int
            Structure KLIFS ID.
        """
        if klifs_session is None:
            klifs_session = setup_remote()

        fingerprint = cls()
        fingerprint.structure_klifs_id = structure_klifs_id

        pocket_bp, pocket_df = fingerprint._get_pocket(structure_klifs_id, klifs_session)
        # Check if residues are consistent between pockets
        if pocket_bp._residue_ids != pocket_df._residue_ids:
            raise ValueError(f"Residue PDB IDs are not the same for df and bp pockets.")
        if pocket_bp._residue_ixs != pocket_df._residue_ixs:
            raise ValueError(f"Residue indices are not the same for df and bp pockets.")
        # Set residue attributes
        fingerprint.residue_ids = pocket_bp._residue_ids
        fingerprint.residue_ixs = pocket_bp._residue_ixs

        values_dict = {}
        values_dict["physicochemical"] = fingerprint._get_physicochemical_features_dict(pocket_bp)
        values_dict["spatial"] = fingerprint._get_spatial_features_dict(pocket_df)
        fingerprint.values_dict = values_dict

        return fingerprint

    def values_array(self, physicochemical=True, spatial_distances=False, spatial_moments=True):
        """TODO Cast dictionaries to 1D numpy array"""

        features = []

        if physicochemical:
            physchem_features = self.values_dict["physicochemical"]
            physchem_features = np.array(list(physchem_features.values())).flatten()
            features.append(physchem_features)

        if spatial_distances:
            distances_features = self.values_dict["spatial"]["distances"]
            distances_features = np.array(list(distances_features.values())).flatten()
            features.append(distances_features)

        if spatial_moments:
            moments_features = self.values_dict["spatial"]["moments"]
            moments_features = np.array(list(moments_features.values())).flatten()
            features.append(moments_features)

        # Concatenate physicochemical and spatial features
        if len(features) > 0:
            features = np.concatenate(features, axis=0)
        else:
            features = np.array([])

        return features

    def _get_pocket(self, structure_klifs_id, klifs_session):
        """TODO"""

        # Set up BioPython-based pocket
        pocket_bp = PocketBioPython.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        # Set up DataFrame-based pocket
        pocket_df = PocketDataFrame.from_structure_klifs_id(
            structure_klifs_id, klifs_session=klifs_session
        )
        return pocket_bp, pocket_df

    def _get_physicochemical_features_dict(self, pocket_bp):
        """TODO"""

        # Set up physicochemical features
        features = {}
        # Add SiteAlign features
        for sitealign_feature_name in ["size", "hbd", "hba", "charge", "aromatic", "aliphatic"]:
            feature = SiteAlignFeature.from_pocket(pocket_bp, "hba")
            features[sitealign_feature_name] = feature.values
        # Add side chain orientation feature
        feature = SideChainOrientationFeature.from_pocket(pocket_bp)
        features["sco"] = feature.values
        # Add solvent exposure feature
        feature = SolventExposureFeature.from_pocket(pocket_bp)
        features["exposure"] = feature.values

        return features

    def _get_spatial_features_dict(self, pocket_df):
        """TODO"""

        # Set up spatial features
        features = {}
        # Add subpockets features
        feature = SubpocketsFeature.from_pocket(pocket_df)
        features["distances"] = feature._distances
        features["moments"] = feature._moments

        return features

    def _normalize_physicochemical_bits(self):
        """
        TODO adapt to new class structure!
        Normalize physicochemical bits.

        Returns
        -------
        pandas.DataFrame
            8 physicochemical features (columns) for 85 residues (rows).
        """

        if self.physicochemical is not None:

            # Make a copy of DataFrame
            normalized = self.physicochemical.copy()

            # Normalize size
            normalized["size"] = normalized["size"].apply(lambda x: self._normalize(x, 1.0, 3.0))

            # Normalize pharmacophoric features: HBD, HBA and charge
            normalized["hbd"] = normalized["hbd"].apply(lambda x: self._normalize(x, 0.0, 3.0))
            normalized["hba"] = normalized["hba"].apply(lambda x: self._normalize(x, 0.0, 2.0))
            normalized["charge"] = normalized["charge"].apply(
                lambda x: self._normalize(x, -1.0, 1.0)
            )

            # No normalization needed for aromatic and aliphatic features which are already 0 or 1

            # Normalize side chain orientation
            normalized["sco"] = normalized["sco"].apply(lambda x: self._normalize(x, 0.0, 2.0))

            # No normalization needed for exposure feature which is already between 0 and 1

            return normalized

        else:
            return None

    def _normalize_distances_bits(self):
        """
        TODO adapt to new class structure!
        Normalize distances bits.

        Returns
        -------
        pandas.DataFrame
            4 distance features (columns) for 85 residues (rows).
        """

        if self.distances is not None:

            # Make a copy of DataFrame
            normalized = self.distances.copy()

            # Normalize using cutoffs defined for each reference point
            normalized["distance_to_centroid"] = normalized["distance_to_centroid"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_centroid"][0],
                    DISTANCE_CUTOFFS["distance_to_centroid"][1],
                )
            )
            normalized["distance_to_hinge_region"] = normalized["distance_to_hinge_region"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_hinge_region"][0],
                    DISTANCE_CUTOFFS["distance_to_hinge_region"][1],
                )
            )
            normalized["distance_to_dfg_region"] = normalized["distance_to_dfg_region"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_dfg_region"][0],
                    DISTANCE_CUTOFFS["distance_to_dfg_region"][1],
                )
            )
            normalized["distance_to_front_pocket"] = normalized["distance_to_front_pocket"].apply(
                lambda x: self._normalize(
                    x,
                    DISTANCE_CUTOFFS["distance_to_front_pocket"][0],
                    DISTANCE_CUTOFFS["distance_to_front_pocket"][1],
                )
            )

            return normalized

        else:
            return None

    def _normalize_moments_bits(self):
        """
        TODO adapt to new class structure!
        Normalize moments bits.

        Returns
        -------
        pandas.DataFrame
            3 moment features (columns) for 4 distance distributions residues (rows).
        """

        if self.moments is not None:

            # Make a copy of DataFrame
            normalized = self.moments.copy()

            # Normalize using cutoffs defined for each moment
            normalized["moment1"] = normalized["moment1"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment1"][0], MOMENT_CUTOFFS["moment1"][1]
                )
            )
            normalized["moment2"] = normalized["moment2"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment2"][0], MOMENT_CUTOFFS["moment2"][1]
                )
            )
            normalized["moment3"] = normalized["moment3"].apply(
                lambda x: self._normalize(
                    x, MOMENT_CUTOFFS["moment3"][0], MOMENT_CUTOFFS["moment3"][1]
                )
            )

            return normalized

        else:
            return None

    @staticmethod
    def _normalize(value, minimum, maximum):
        """
        TODO adapt to new class structure!
        Normalize a value using minimum-maximum normalization.
        Values equal or lower / greater than the minimum / maximum value are set to 0.0 / 1.0.

        Parameters
        ----------
        value : float or int
            Value to be normalized.
        minimum : float or int
            Minimum value for normalization, values equal/greater than this minimum are set to 0.0.
        maximum : float or int
            Maximum value for normalization, values equal/greater than this maximum are set to 1.0.

        Returns
        -------
        float
            Normalized value.
        """

        if minimum < value < maximum:
            return (value - minimum) / float(maximum - minimum)
        elif value <= minimum:
            return 0.0
        elif value >= maximum:
            return 1.0
        elif np.isnan(value):
            return np.nan
        else:
            raise ValueError(f"Unexpected value to be normalized: {value}")
